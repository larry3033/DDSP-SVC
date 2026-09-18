import torch
import torch.nn as nn


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]
    return signal.permute(0, 2, 1)
