import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

from utils import gen_sub


class ProbaVDatasetTrain(torch.utils.data.Dataset):
    """multitemporal scenes."""

    def __init__(self, config):

        x_LR = np.load(config.train_lr_file).astype(np.float32)
        x_HR = np.load(config.train_hr_file).astype(np.float32)
        M = np.load(config.train_masks_file).astype(np.float32)

        x_LR = x_LR[:config.max_train_scenes]
        x_HR = x_HR[:config.max_train_scenes]
        M = M[:config.max_train_scenes]

        self.x_LR_patches = gen_sub(x_LR, config.patch_size, config.patch_size)
        self.x_HR_patches = gen_sub(x_HR, config.patch_size*3, config.patch_size*3)
        self.M_patches = gen_sub(M, config.patch_size*3, config.patch_size*3)

        valid_pos = np.mean(self.M_patches,(1,2,3))>0.1 # extra clearance check on patches
        self.x_LR_patches = self.x_LR_patches[valid_pos]
        self.x_HR_patches = self.x_HR_patches[valid_pos]
        self.M_patches = self.M_patches[valid_pos]

        self.mu = 7433.6436
        self.sigma = 2353.0723


    def __len__(self):
        return len(self.x_LR_patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR_patches[idx] - self.mu) / self.sigma
        x_HR = self.x_HR_patches[idx]
        M = self.M_patches[idx]

        return np.transpose(x_LR,(2,0,1)).astype(np.float32), np.transpose(x_HR,(2,0,1)).astype(np.float32), np.transpose(M,(2,0,1)).astype(np.float32) # (T,X,Y)



class ProbaVDatasetVal(torch.utils.data.Dataset):
    """multitemporal scenes."""

    def __init__(self, config):

        self.x_LR = np.load(config.val_lr_file).astype(np.float32)
        self.x_HR = np.load(config.val_hr_file).astype(np.float32)
        self.M = np.load(config.val_masks_file).astype(np.float32)

        self.mu = 7433.6436
        self.sigma = 2353.0723

    def __len__(self):
        return len(self.x_LR)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_LR = (self.x_LR[idx] - self.mu) / self.sigma
        x_HR = self.x_HR[idx]
        M = self.M[idx]

        return np.transpose(x_LR,(2,0,1)).astype(np.float32), np.transpose(x_HR,(2,0,1)).astype(np.float32), np.transpose(M,(2,0,1)).astype(np.float32)