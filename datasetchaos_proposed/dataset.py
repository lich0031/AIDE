import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import pydicom
import torch

palette = [[0], [63], [126], [189], [252]]

class chaos_seg(Dataset):

    def __init__(self, root, csv_file, tempmaskfolder, train=True, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.t1inphase = img_mask['Inphase'].values.tolist()
        self.t1outphase = img_mask['Outphase'].values.tolist()
        self.masks = img_mask['Mask'].values.tolist()
        self.tempmaskfolder = tempmaskfolder
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        inphaseimg = pydicom.read_file(os.path.join(self.root, self.t1inphase[idx])).pixel_array
        inphaseimg = Image.fromarray(inphaseimg)
        if inphaseimg.mode != 'RGB':
            inphaseimg = inphaseimg.convert('RGB')

        outphaseimg = pydicom.read_file(os.path.join(self.root, self.t1outphase[idx])).pixel_array
        outphaseimg = Image.fromarray(outphaseimg)
        if outphaseimg.mode != 'RGB':
            outphaseimg = outphaseimg.convert('RGB')
        caseidx = self.t1inphase[idx].split('/')[2]
        if not caseidx.isdigit():
            caseidx = self.t1inphase[idx].split('/')[0]
        maskpath = os.path.join(self.root, self.masks[idx])
        if self.train:
            maskpath1 = os.path.join(self.root, self.tempmaskfolder, str(caseidx),
                                     self.masks[idx].split('/')[-1].split('.')[0]+'_net1.png')
            maskpath2 = os.path.join(self.root, self.tempmaskfolder, str(caseidx),
                                     self.masks[idx].split('/')[-1].split('.')[0]+'_net2.png')
        else:
            maskpath1 = maskpath
            maskpath2 = maskpath

        mask = Image.open(maskpath)

        if os.path.exists(maskpath1):
            mask1 = Image.open(maskpath1)
        else:
            mask1 = Image.open(maskpath)

        if os.path.exists(maskpath2):
            mask2 = Image.open(maskpath2)
        else:
            mask2 = Image.open(maskpath)

        if mask.mode != 'L':
            mask = mask.convert('L')
        if mask1.mode != 'L':
            mask1 = mask1.convert('L')
        if mask2.mode != 'L':
            mask2 = mask2.convert('L')

        augset = {
            'augno': 4,
            'imgmodal11': inphaseimg.copy(),
            'imgmodal21': outphaseimg.copy(),
            'degree1': 0.0,
            'hflip1': 0,
            'imgmodal12': inphaseimg.copy(),
            'imgmodal22': outphaseimg.copy(),
            'degree2': 0.0,
            'hflip2': 0,
            'imgmodal13': inphaseimg.copy(),
            'imgmodal23': outphaseimg.copy(),
            'degree3': 0.0,
            'hflip3': 0,
            'imgmodal14': inphaseimg.copy(),
            'imgmodal24': outphaseimg.copy(),
            'degree4': 0.0,
            'hflip4': 0,
        }

        if self.transform:
            inphaseimg, outphaseimg, augset, mask, mask1, mask2 = self.transform(inphaseimg, outphaseimg,
                                                                                 augset, mask, mask1, mask2)

        mask_arr = np.array(mask)
        mask_arr = np.expand_dims(mask_arr, axis=2)
        mask_arr = one_hot_mask(mask_arr, palette)
        mask = mask_arr.transpose([2, 0, 1])
        mask = torch.from_numpy(mask).long()

        mask_arr = np.array(mask1)
        mask_arr = np.expand_dims(mask_arr, axis=2)
        mask_arr = one_hot_mask(mask_arr, palette)
        mask1 = mask_arr.transpose([2,0,1])
        mask1 = torch.from_numpy(mask1).long()

        mask_arr = np.array(mask2)
        mask_arr = np.expand_dims(mask_arr, axis=2)
        mask_arr = one_hot_mask(mask_arr, palette)
        mask2 = mask_arr.transpose([2,0,1])
        mask2 = torch.from_numpy(mask2).long()

        return inphaseimg, outphaseimg, augset, mask, mask1, mask2

    def __len__(self):
        return len(self.masks)

def one_hot_mask(label, label_values):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map.astype(np.uint8))
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map