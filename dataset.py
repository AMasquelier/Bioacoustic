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
from sklearn.model_selection import StratifiedKFold

from utils import set_seed



DATA_VERSIONS = {
    # 'clean1':['cleaned_audio', 'train_clean.csv'],
    'clean1':['cleaned_audio_1_1', 'train_clean_1_1.csv'],
    'clean2':['cleaned_audio2', 'train_clean2.csv'],
    'processed':['processed_audio_2', 'train_processed2.csv'],
}
    

class BirdCLEF2026Dataset(Dataset):
    PATH = '../../Datasets/BirdCLEF/2026/'
    
    config = {"padding":"random", "sampling":"energy", "sr":32000, 'seed':2, 'train_only':False, 'extra_data':False, 'soundscape_data':True, 'synonymous_taxa':True, 'sampling_gamma':0.5, 'batch_size':32, 'num_workers':0, 'data_version':'processed'}

    def get_site(self, x):
        try:
            x = x.split('_')
            site = x[-5]
        except:
            site = None
        return site
    
    def __init__(self, split='train', fold=0, config={}):
        self.config.update(config)

        tax = pd.read_csv(self.PATH+'taxonomy.csv')
        self.split = split

        self.SND_PATH = self.PATH + f'{DATA_VERSIONS[self.config['data_version']][0]}/'
        self.LABELS = list(np.unique(tax.primary_label.dropna()))

        df = pd.read_csv(self.PATH+DATA_VERSIONS[self.config['data_version']][1])
        df['secondary_labels'] = df['secondary_labels'].apply(eval)
        df['filename'] = self.SND_PATH+df['filename']
        df['src'] = 'original'

        no_call = pd.read_csv(self.PATH+f'no_call.csv')
        no_call['filename'] = self.PATH+'train_soundscapes/'+no_call['filename']+'.ogg'
        no_call['primary_label'] = no_call.filename.apply(lambda x: [])
        no_call['secondary_labels'] = no_call.filename.apply(lambda x: [])
        no_call['hash'] = no_call['filename'].values # to change
        no_call['src'] = 'no_call'

        syndf = pd.read_csv(self.PATH+f'synonyms_data_clean.csv')
        syndf['primary_label'] = syndf['primary_label'].apply(lambda x: [x])
        syndf = syndf[(syndf.status=='OK') & (syndf.hash>0)]
        syndf['secondary_labels'] = syndf['secondary_labels'].apply(eval)
        syndf['src'] = 'synonym'

        ssdf = pd.read_csv(self.PATH+f'train_soundscapes_clean.csv')
        ssdf['primary_label'] = ssdf['primary_label'].apply(eval)
        ssdf = ssdf[ssdf.status=='OK']
        ssdf['secondary_labels'] = '[]'
        ssdf['secondary_labels'] = ssdf['secondary_labels'].apply(eval)
        ssdf['filename'] = self.PATH+'cleaned_audio/'+ssdf['filename']
        ssdf['src'] = 'soundscapes'

        
        df = pd.concat([df, ssdf, syndf, no_call])
        df = df.drop_duplicates(subset=['hash'], keep='first').drop_duplicates(subset=['filename'], keep='first')
        df.index = df.filename.values

        self.cp = df.copy()

        
        IDX = np.unique(df[df['src']=='original'].index)
        set_seed(self.config['seed'])
        np.random.shuffle(IDX)

        if self.config['train_only']: idx = IDX
        else:
            skf = StratifiedKFold(n_splits=5)
            FOLDS = list(skf.split(IDX, df.loc[IDX].primary_label.astype(str).fillna('none').values))
            train_idx = IDX[FOLDS[fold][0]].tolist()
            val_idx = IDX[FOLDS[fold][1]].tolist()
            idx = train_idx if self.split=='train' else val_idx
            
        DF = df.loc[idx].copy()

        counts = DF['primary_label'].value_counts()
        class_weights = (counts / counts.sum())**(-self.config['sampling_gamma'])
        DF['weights'] = DF.primary_label.map(class_weights).values
        

        DF['primary_label'] = DF['primary_label'].apply(lambda x: [x])

        DF = pd.concat([DF, df[df.src=='no_call']])
        
        if self.split=='train' and self.config['extra_data']:
            DF = pd.concat([DF, df[df.src=='additional']])

        if self.config['soundscape_data']:
            tmp = df[df.src=='soundscapes']
            if not self.config['train_only']:
                tmp['site'] = tmp.filename.apply(self.get_site)
                tmp['recording_id'] = tmp.filename.apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-2]))
                tmp.index = tmp.recording_id.values
    
                if not(self.config['train_only']):
                    s22 = tmp[(tmp.site=='S22')]
                    s08 = tmp[(tmp.site=='S08')]
                    tmp = tmp[(tmp.site!='S22')&(tmp.site!='S08')]
        
                    
                    s22_idx = np.unique(s22.index.values.copy())
                    s08_idx = np.unique(s08.index.values.copy())
                    np.random.shuffle(s22_idx)
                    s22_idx = np.roll(s22_idx, fold*int(0.3*len(s22_idx)))
                    
                    if self.split=='train':
                        tmp = tmp[(tmp.site!='S03') & (tmp.site!='S18')]
                        tmp = pd.concat([tmp, s22.loc[s22_idx[:int(0.7*len(s22_idx))]], s08.loc[s08_idx[1:]]])
                    else:
                        tmp = tmp[(tmp.site=='S03') | (tmp.site=='S18')]
                        tmp = pd.concat([tmp, s22.loc[s22_idx[int(0.7*len(s22_idx)):]], s08.loc[s08_idx[[0]]]])

            self.ss_df = tmp.copy()
            DF = pd.concat([DF, tmp])
            

        if self.split=='train' and self.config['synonymous_taxa']:
            DF = pd.concat([DF, df[df.src=='synonym']])

        DF['weights'] = DF['weights'].fillna(np.mean(class_weights))
        self.DF = DF
        

        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=DF['weights'].values,
            num_samples=len(DF['weights']),
            replacement=True
        )


        self.paths = list(DF['filename'].values)
        labels = DF['primary_label'].apply(self.make_labels).values
        secondary_labels = DF['secondary_labels'].apply(self.make_labels).values
        self.labels = (labels | secondary_labels)#.astype(int)
        self.src_weights = torch.ones(234)
        self.src_weights[tax[tax.primary_label.str.contains('47158son')].index] = 0.1

        

    def get_loader(self):
        if self.split=='train':
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=False,
                drop_last=True,
                sampler=self.sampler,
                multiprocessing_context="fork",
            )
        else:
            return DataLoader(
                self,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=False,
                multiprocessing_context="fork",
            )

        
    def make_labels(self, X):
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
                
        return wav.clone()
        

    def __len__(self):
        return len(self.paths)
        

    def __getitem__(self, idx):
        
        path = self.paths[idx]
        DUR = int(5 * self.config['sr'])

        audio = self.load_sound(path, DUR=DUR)
        labels = self.labels[idx]
        weights = torch.ones(234) if 'BC2026' in path else self.src_weights
        
        return (
                audio,
                torch.tensor(labels, dtype=torch.float32)
            )


            


class XCDataset(Dataset):
    PATH = "../../Datasets/Xeno-Canto/"
    
    DEFAULT_CONFIG = {"padding":"random", "sr":32000, 'seed':2, 'train_only':False, 'target':'sci_name'}

    def __init__(self, split='train', config={}):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config.update(config)
        

        df = pd.read_csv(self.PATH+'files_details.csv', low_memory=False)

        # Filtering
        df['license'] = df.lic.apply(lambda x: x.split('/')[-3])
        df = df[df.sp!='mystery']
        df = df[(df.license!='by-nc-nd') & (df.license!='3.0')]
        
        df['filename'] = self.PATH+'audio/'+df['sp'].astype(str)+'/XC'+df['id'].astype(str)+'.ogg'
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




class SoundscapeDataset(Dataset):
    # Uses soundscape datasets listed at https://huggingface.co/datasets/DBD-research-group/BirdSet
    PATH = "../../Datasets/Soundscapes/"
    
    DEFAULT_CONFIG = {"padding":"random", "sr":32000, 'seed':2, 'train_only':False, 'target':'sci_name'}

    def __init__(self, LABELS, dataset='PER', split='train', config={}):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config.update(config)

        mapping = pd.read_csv('../../Datasets/Soundscapes/eBird_taxonomy.csv')
        mapping.index = mapping.SPECIES_CODE.values
        self.mapping = {sp:mapping.loc[sp,'SCI_NAME'] for sp in mapping.index}

        df = pd.read_csv(f"{self.PATH}{dataset}/clip_annotations.csv", low_memory=False)
        df['labels'] = df['labels'].apply(eval)

        df = df[df.n_labels>0].copy()

        ##################################
        #  No splits - validation only   #
        ##################################
        
        self.LABELS = LABELS
        DF = df.copy()

        self.paths = list(DF['filename'].values)
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
            
        tax = pd.read_csv(path+'eBird_taxonomy.csv')
        tax = tax.rename(columns={c:c.lower() for c in tax.columns})
        df = pd.read_csv(f"{path}{dataset}/annotations.csv")
        df = df.rename(columns={
            'Species eBird Code':'species_code', 
            'Start Time (s)':'start', 
            'End Time (s)':'end'
        })

        files = df.Filename.unique()
        
        os.makedirs(f"{path}{dataset}/soundscape_clips", exist_ok=True)

        annotations = pd.DataFrame(columns=['filename', 'start', 'labels', 'n_labels'])
        
        for filename in tqdm(files):
            wav, sr = torchaudio.load(f"{path}{dataset}/soundscape_data/{filename}")
            L = wav.shape[1]//sr
    
            for start in range(0,L-5,5):
                start = sr * start
                duration = sr * 5
                
                clip = wav[:,start:start+duration]
            
                sub = df[df.Filename==filename]
                sub = sub[(sub.end*sr > start) & (sub.start*sr < start+duration)]
        
                new_filename = f"{path}{dataset}/soundscape_clips/{filename.split('.flac')[0]}_{start//sr}.ogg"
                annotations.loc[len(annotations)] = [new_filename, start//sr, list(sub.species_code.unique()), len(sub.species_code.unique())]
        
                torchaudio.save(new_filename, src=clip, sample_rate=32000)
        
            annotations.to_csv(f"{path}{dataset}/clip_annotations.csv", index=False)