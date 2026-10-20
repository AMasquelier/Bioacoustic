import torch.nn as nn
import torchaudio
import torch

import numpy as np

class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512, win_length=640, f_min=20, f_max=16000, channels=1, norm="slaney", mel_scale="htk", target_size=(256, 256), top_db=80.0, delta_win=5,):
        super().__init__()
        self.channels = channels
        self.top_db = top_db

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            mel_scale=mel_scale,
            pad_mode="reflect",
            power=2.0,
            norm=norm,
            center=True,
            win_length=win_length,
        )

        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

    def power_to_db(self, S):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(S)) 
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x):
        # x: (B, T) or (T,)
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        mel_spec = self.mel_transform(x)          # (B, n_mels, time)
        mel_spec = self.power_to_db(mel_spec)[:,:,:-1]

        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)

        B, C = mel_spec.shape[:2]
        flat = mel_spec.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)

        if squeeze:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec

    def m(self, f):
        return 2595 * np.log10(1+(f/700))

    def time_to_coord(self, t, L):
        SL = np.floor(L / self.hop_length)+1
        return t * SL

    def freq_to_coord(self, f):
        return (m(f)-m(self.f_min))/(m(self.f_max)-m(self.f_min))*self.n_mels
        