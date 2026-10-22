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



class SoundscapeDataset(Dataset):
    # Uses soundscape datasets listed at https://huggingface.co/datasets/DBD-research-group/BirdSet
    PATH = "../../Datasets/Soundscapes/"
    
    DEFAULT_CONFIG = {"padding":"random", "sr":32000, 'seed':2, 'train_only':False, 'target':'sci_name'}

    def __init__(self, LABELS, dataset='PER', split='train', config={}):
        PATH = Path(self.PATH)
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config.update(config)

        mapping = pd.read_csv(PATH / dataset / "species.csv")
        mapping.index = mapping['Species eBird Code'].values
        self.mapping = {sp:mapping.loc[sp,'Scientific Name'] for sp in mapping.index}

        df = pd.read_csv(PATH / dataset / "clip_annotations.csv", low_memory=False)
        df['labels'] = df['labels'].apply(eval)

        df = df[df.n_labels>0].copy()

        ##################################
        #  No splits - validation only   #
        ##################################
        
        self.LABELS = LABELS
        DF = df.copy()

        self.paths = list(PATH / dataset / "soundscape_clips" / DF['filename'].values)
        self.targets = DF['labels'].apply(self.make_labels).values

        self.dataset = dataset
        self.split = split
        self.DF = DF
        
        

    def make_labels(self, X):
        out = np.zeros(len(self.LABELS)).astype(bool)
        for x in X:
            if x in self.mapping:
                x = self.mapping[x]
                if x in self.LABELS:
                    out[self.LABELS.index(x)] = True
            
        return out
        

    def load_sound(self, filepath):
        wav, sr = torchaudio.load(filepath)
        wav = wav[0]
                
        return wav

        

    def get_loader(self):
            
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

        audio = self.load_sound(path)
        targets = self.targets[idx]
        
        return (
            audio,
            torch.tensor(targets, dtype=torch.float32)
        )

    def __repr__(self):
        return f"SoundscapeDataset [{self.dataset}]"


    def process_dataset(dataset, path="../../Datasets/Soundscapes/"):
        path = Path(path) 
        
        df = pd.read_csv(path / dataset / "annotations.csv")
        df = df.rename(columns={
            'Species eBird Code':'species_code', 
            'Start Time (s)':'start', 
            'End Time (s)':'end'
        })

        files = df.Filename.unique()
        
        os.makedirs(path / dataset / "soundscape_clips", exist_ok=True)

        annotations = pd.DataFrame(columns=['filename', 'start', 'labels', 'n_labels'])
        
        for filename in tqdm(files):
            wav, sr = torchaudio.load(path / dataset / f"soundscape_data/{filename}")
            L = wav.shape[1]//sr
    
            for start in range(0,L-5,5):
                start = sr * start
                duration = sr * 5
                
                clip = wav[:,start:start+duration]
            
                sub = df[df.Filename==filename]
                sub = sub[(sub.end*sr > start) & (sub.start*sr < start+duration)]
        
                new_filename = f"{filename.split('.flac')[0]}_{start//sr}.ogg"
                annotations.loc[len(annotations)] = [new_filename, start//sr, list(sub.species_code.unique()), len(sub.species_code.unique())]
        
                torchaudio.save(path / dataset / "soundscape_clips" / new_filename, src=clip, sample_rate=32000)
        
            annotations.to_csv(path / dataset / "clip_annotations.csv", index=False)