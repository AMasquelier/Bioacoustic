import numpy as np
import pandas as pd
import torch.nn as nn
import torch


class CASL(nn.Module):
    def __init__(self, gamma=(2,0), margin=0.05, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.margin = margin

        pos_gamma = torch.tensor(gamma[1], dtype=torch.float32)
        neg_gamma = torch.tensor(gamma[0], dtype=torch.float32)        

        self.register_buffer('pos_gamma', pos_gamma)
        self.register_buffer('neg_gamma', neg_gamma)
    
    def forward(self, logits, y, reduction=None):
        if reduction is None: reduction = self.reduction
        
        eps = 1e-7
        pos_gamma = self.pos_gamma.unsqueeze(0).to(y.device)
        neg_gamma = self.neg_gamma.unsqueeze(0).to(y.device)

        y = y.float()
        p = logits.float().sigmoid()
        p = torch.clamp(p, eps, 1-eps)        

        p_m = torch.maximum(torch.tensor(eps), p-self.margin)
        

        l_1 = -((1-p)**pos_gamma)*torch.log(p)
        l_0 = -(p_m**neg_gamma)*torch.log(1-p_m)
        
        asl = y * l_1 + (1-y) * l_0

        if reduction=='mean': asl = asl.mean()
            
        return asl

    def __repr__(self):
        return f'CASL - pos_gamma={str(self.pos_gamma)} - pos_gamma={str(self.neg_gamma)} - margin={self.margin}' 
        

class ASL(nn.Module):
    name = 'ASL'
    def __init__(self, gamma=(2,0.), margin=0.05, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.margin = margin
    
    def forward(self, logits, y, reduction=None):
        if reduction is None: reduction = self.reduction

        reduction = reduction or self.reduction

        eps = 1e-7
        y = y.float()
        p = logits.float().sigmoid()
        p = torch.clamp(p, eps, 1-eps)

        p_m = torch.maximum(torch.tensor(eps), p-self.margin)

        l_1 = -((1-p)**self.gamma[1])*torch.log(p)
        l_0 = -(p_m**self.gamma[0])*torch.log(1-p_m)
        
        asl = y * l_1 + (1-y) * l_0

        if reduction=='mean': asl = asl.mean()
            
        return asl

    def __repr__(self):
        return f'ASL - gamma={str(self.gamma)} - margin={self.margin}' 
        

class Focal_BCE(nn.Module):
    name = 'Focal_BCE'
    def __init__(self, alpha=0, gamma=2.0, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, reduction='mean'):
        
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")

        prob = pred.sigmoid()
        pt = target * prob + (1 - target) * (1 - prob)
        loss = bce * (1 - pt).pow(self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if reduction=='mean': loss = torch.mean(loss)

        return loss

    def __repr__(self):
        return f'FocalBCE - gamma={str(self.gamma)} - alpha={self.alpha}' 
        

class BCE(nn.Module):
    name = 'BCE'
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits, target, reduction='mean'):
        if reduction is None: reduction = self.reduction

        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction=reduction)
        return loss

    def __repr__(self):
        return f'BCE' 