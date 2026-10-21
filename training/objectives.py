import torch
import torchvision
import torch.nn as nn

from augmentations import Masking, Stretch, MixupAudio
from spectrogram import Spectrogram
from metrics import AUC
from losses import BCE


class Baseline(nn.Module):
    logged_losses = ('train_loss',)
    
    def __init__(self, model, mel, aug, device, config=None):
        super().__init__()
        self.config = {'loss':BCE(), 'metrics':[AUC], 'seed':2, 'seed_mod':2, 'fold':0, 'augmentation':None, 'soundscape_dataset':[]}
        if config: self.config.update(config)

        self.device = device
        self.use_amp = (self.device.type == 'cuda')

        self.mixup = MixupAudio(**self.config['mix'])
        self.loss_fn = self.config['loss']
        self.model = model
        self.aug = aug
        self.mel = mel
        
    
    def forward(self, x, y, lam=0):
        device = x.device
        x, y = self.mixup(x, y)
        x = self.aug(self.mel(x))
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits = self.model(x)
            L = self.loss_fn(logits, y)
            
        return {"train_loss" : L}

    def __repr__(self):
        return f'Baseline'



        
class MixupConsistency(nn.Module):
    logged_losses = ('train_loss', 'train_loss_LS', 'train_loss_LM', 'train_loss_LC')
    
    def __init__(self, config=None):
        super().__init__()
        self.config = {'loss':BCE(), 'metrics':[AUC], 'seed':2, 'seed_mod':2, 'fold':0, 'augmentation':None, 'soundscape_dataset':[]}
        if config: self.config.update(config)

        self.device = device
        self.use_amp = (self.device.type == 'cuda')

        self.mixup = MixupAudio(**self.config['mix'])
        self.loss_fn = self.config['loss']
        self.model = model
        self.aug = aug
        self.mel = mel
        
    
    def forward(self, x, y, lam=0):
        x_single = self.aug(self.mel(x))

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits_single = self.model(x_single)

        x_mix, (y_mix,logits_single_mix) = self.mixup(x, [y,logits_single.float()])

        x_mix = self.aug(self.mel(x_mix))

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits_mix = model(x_mix)

            y_single_mix = logits_single_mix.sigmoid().detach()#(logits_single_mix.sigmoid().detach()>0.6).float()
            
            LS = self.loss_fn(logits_single, y, reduction='mean')
            LM = self.loss_fn(logits_mix, y_mix, reduction='mean')
            LC = self.loss_fn(logits_mix, y_single_mix, reduction='mean')
            L = LS + LM + self.config['consistency_schedule'](lam) * LC
            
        return {
            'train_loss':L,
            'train_loss_LS':LS,
            'train_loss_LM':LM,
            'train_loss_LC':LC,
        }

    def __repr__(self):
        return f'MixupConsistency'

