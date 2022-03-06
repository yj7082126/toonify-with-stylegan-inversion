import copy
import numpy as np
from PIL import Image
import torch

def preproc(x):
    x = np.clip(np.asarray(x) / 255., 0.0, 1.0)
    x = torch.from_numpy(x).permute(2,0,1)*2-1
    return x.float().unsqueeze(0)

def postproc(x):
    x = ((x.permute(1,2,0)+1)/2).clamp(0, 1)
    x = (x.cpu().numpy() * 255.).astype(np.uint8)
    x = Image.fromarray(x, 'RGB')
    return x