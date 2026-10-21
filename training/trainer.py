import hashlib
import datetime
import numpy as np
from time import time
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from IPython.display import clear_output

from augmentations import Masking, Stretch, MixupAudio
from dataset import SoundscapeDataset
from spectrogram import Spectrogram
from utils import set_seed
from metrics import AUC
from losses import BCE
from utils import Log

from training.objectives import Baseline


class Trainer:
    
    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {'loss':BCE(), 'metrics':[AUC], 'seed':2, 'seed_mod':2, 'fold':0, 'augmentation':None, 'soundscape_dataset':[], 'objective':Baseline}
        if config: self.config.update(config)
        
        if self.config['augmentation']: self.aug = self.config['augmentation']
        else:
            self.aug = torchvision.transforms.Compose([
                            Stretch(.1, axis='x'),
                            Masking(N=8, max_freq_band=2, max_time_band=3, shape='cross')
                        ])

        self.exp_id = hashlib.sha256(str(time()).encode()).hexdigest()

        self.mel = Spectrogram(**self.config['mel']).to(self.device)

        set_seed(self.config['seed'])
        train_dataset = self.config['dataset']('train', config=self.config)
        val_dataset  = self.config['dataset']('val', config=self.config)

        self.train_loader = train_dataset.get_loader()
        self.val_loader = val_dataset.get_loader()

        self.soundscapes_loader = {}
        for ssds in self.config['soundscape_dataset']:
            soundscape_dataset = SoundscapeDataset(LABELS=train_dataset.LABELS, dataset=ssds, config=self.config)
            self.soundscapes_loader[ssds] = soundscape_dataset.get_loader()
            
        self.validation = {'Validation':self.val_loader}|self.soundscapes_loader
        
        self.model = self.config['model'](config=self.config).to(self.device)
        self.loss_fn = self.config['loss']
        self.metrics = self.config['metrics']

        self.objective = self.config['objective'](self.model, self.mel, self.aug, self.device, config=self.config)

        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            

    def train_one_epoch(self, epoch, verbose=1):
        self.model.train()        
        Loss = {l : 0 for l in self.objective.logged_losses}
        n_steps = len(self.train_loader)
        total_steps = self.epochs * n_steps

        if verbose==2: pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.train_loader)

        lam = epoch / self.epochs
        for batch_idx, (x, y) in pbar:
            self.optimizer.zero_grad()

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            batch_size = x.size(0)

            L = self.objective(x, y, lam)

            self.scaler.scale(L['train_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for l in L:
                Loss[l] += L[l].detach().item()

        
        log = {l:Loss[l] / n_steps for l in Loss}

        return log

        
                
    def validate(self, verbose=1):
        self.model.eval()

        logs = {}
        for val_set in self.validation:
            loader = self.validation[val_set]
            log = {}
            Loss = 0
    
            n_steps = len(loader)
    
            if verbose==2: pbar = tqdm(enumerate(loader), total=n_steps, desc=f"Validation: {val_set}")
            else: pbar = enumerate(loader)
    
            preds = []
            targets = []
    
            with torch.no_grad():
                for batch_idx, (x, y) in pbar:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
    
                    x = self.mel(x)

                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = self.model(x)
                        loss = self.loss_fn(logits, y)
    
                    Loss += loss.detach().item()
                    preds.append(logits.float().sigmoid().cpu())
                    targets.append(y.cpu())
    
            preds = torch.cat(preds).numpy()
            targets = torch.cat(targets).numpy()
    
            log[f'{val_set}_loss'] = Loss / n_steps
                   
            for m in self.metrics:
                log[f'{val_set}_{m.__name__}'] = m(targets, preds)

            logs[val_set] = log

        return logs

    
    def train(self, epochs=16, verbose=1, checkpoint_freq='latest'):
        set_seed(self.config['seed'])

        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-8)

        log = Log()

        best_loss = np.inf
        
        Epochs = range(epochs)

        t0 = time()
        for epoch in Epochs:
            set_seed(self.config['seed']+self.config['seed_mod']+epoch+self.config['fold']*256)
            train_log = self.train_one_epoch(epoch,verbose=verbose)
        
            val_log = self.validate(verbose=verbose)

            if checkpoint_freq=='best' and val_log['Validation']['Validation_loss'] < best_loss:
                best_loss = val_log['Validation']['Validation_loss']
                torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")
            
            clear_output(wait=True)
            print(self.exp_id)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f'\033[1m Training \t|\t {'  -  '.join([f'{e}={np.round(train_log[e],3)}' for e in train_log])} \033[0m')
            for val_set in val_log:
                print(f'\033[1m {val_set} \t|\t {'  -  '.join([f'{e}={np.round(val_log[val_set][e],3)}' for e in val_log[val_set]])} \033[0m')
            print()
            print(f"{np.round((time()-t0)/(epoch+1), 2)} s / epoch")

            val_log_ = {}
            for val_set in val_log: val_log_ = val_log_|val_log[val_set]

            metadata = {"timestamp":str(datetime.datetime.now())}
            log.log(epoch, train_log|val_log_|self.config|metadata)
            log.df.to_csv(f'history/{self.exp_id}.csv', index=False)

            if self.scheduler: self.scheduler.step()

            if checkpoint_freq=='latest': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")
                
        if checkpoint_freq=='once': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")

