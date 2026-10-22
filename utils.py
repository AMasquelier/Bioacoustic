import os
import shutil
import numpy as np
import pandas as pd
import time
import hashlib
import torch
import random
import math

from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from torch.optim.lr_scheduler import LambdaLR


os.makedirs('models', exist_ok=True)
os.makedirs('history', exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_config(cfg):
    for X in cfg:
        try: cfg[X] = eval(cfg[X])
        except:None
    return cfg


class Log:
    def __init__(self):
        self.df = pd.DataFrame()

    def log(self, index, column, value):
        self.df.loc[index, col] = value

    def log(self, index, entries):
        for e in entries:
            self.df.loc[index, e] = repr(entries[e])

    def save(self, filename):
        self.df.to_csv(filename, index=False)
    


class LinearSchedule:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * x + self.b

    def __str__(self):
        return f"{self.a} x + {self.b}"