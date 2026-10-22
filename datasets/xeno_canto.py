import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from utils import set_seed



class XCDataset(Dataset):
    PATH = "../../Datasets/Xeno-Canto/"
    
    DEFAULT_CONFIG = {"padding":"random", "sr":32000, 'seed':2, 'train_only':False, 'target':'sci_name', 'fold':0}

    def __init__(self, split='train', config={}):
        PATH = Path(self.PATH)
        
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config.update(config)
        

        df = pd.read_csv(PATH / 'files_details.csv', low_memory=False)

        # Filtering
        df['license'] = df.lic.apply(lambda x: x.split('/')[-3])
        df = df[df.sp!='mystery']
        df = df[(df.license!='by-nc-nd') & (df.license!='3.0')]
        
        df['filename'] = PATH / 'audio' / df['sp'].astype(str) / ('XC'+df['id'].astype(str)+'.ogg')
        df['status'] = df.filename.apply(lambda x: 'OK' if os.path.exists(x) else None)
        df = df.dropna(subset=['status']).drop_duplicates(subset=['filename', 'sp', 'gen'])

        df['sci_name'] = df.gen + ' ' + df.sp

        df = df[df.sci_name!='Sonus naturalis']
        df = df[(df.sci_name!='Vulpes vulpes') | (df.grp!='birds')]

        df['primary_label'] = df[self.config['target']].apply(lambda x: x).values

        # Secondary labels are the equivalent of sci_name
        df['secondary_labels'] = df['also'].apply(eval)
        df['secondary_labels'] = df['secondary_labels'].apply(lambda x: [y.split(' ')[0] for y in x])

        df['label_count'] = df.groupby('primary_label')['primary_label'].transform('size')

        single = df[df.label_count==1].copy()
        multi = df[df.label_count>1].copy()
        
        self.LABELS = list(np.unique(df.primary_label))

        multi.index = multi.filename.values
        
        IDX = np.unique(multi.index)
        set_seed(self.config['seed'])

        if self.config['train_only']: idx = IDX
        else:
            skf = StratifiedKFold(n_splits=20)
            FOLDS = list(skf.split(IDX, multi.loc[IDX].primary_label.fillna('none').values))
            train_idx = IDX[FOLDS[self.config['fold']][0]].tolist()
            val_idx = IDX[FOLDS[self.config['fold']][1]].tolist()
            idx = train_idx if split=='train' else val_idx
            
        DF = multi.loc[idx].copy()

        if split=='train': DF = pd.concat([DF, single])

        self.paths = list(DF['filename'].values)
        targets = DF['primary_label'].apply(self.make_labels).values
        secondary_targets = DF['secondary_labels'].apply(self.make_secondary_labels).values
        self.targets = np.stack((targets | secondary_targets))
        
        self.split = split
        self.DF = DF
        

    def make_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        out[self.LABELS.index(X)] = True
            
        return out

    def make_secondary_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        for x in X:
            if x in self.LABELS:
                out[self.LABELS.index(x)] = True
            
        return out
        

    def load_sound(self, filepath, start=0, DUR=5*32000):
        wav, sr = torchaudio.load(filepath)
        wav = wav[0]

        l = len(wav)
        if l < DUR:
            padding = self.config['padding']
            if padding=='random': padding = np.random.choice(['cycle', 'zero'])
                
            if padding=='cycle':
                n_repeat = int(np.ceil(DUR/l))
                wav = torch.roll(wav.repeat(n_repeat), np.random.randint(l))[:DUR]
            else:
                wav2 = torch.zeros((DUR))
                s = np.random.randint(DUR-l)
                wav2[s:s+l] = wav
                wav = wav2
        else:
            if self.split=='train':
                s = random.randint(0, l-DUR)
                wav = wav[s:s+DUR]
            else:
                wav = wav[:DUR]
                
        return wav

    def get_loader(self):
        if self.split=='train':
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                prefetch_factor=2, 
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                multiprocessing_context="fork",
            )
        else:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                prefetch_factor=2, 
                pin_memory=True,
                shuffle=False,              
                multiprocessing_context="fork",
            )

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, idx):
        path = self.paths[idx]
        DUR = int(self.config['duration'] * self.config['sr'])

        audio = self.load_sound(path, DUR=DUR)
        targets = self.targets[idx]
        
        return (
            audio,
            torch.tensor(targets, dtype=torch.float32)
        )

    def __repr__(self):
        return "XCDataset"