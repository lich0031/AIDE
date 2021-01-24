import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import pydicom
import torch
import SimpleITK as sitk

class breast_seg(Dataset):

    def __init__(self, root, csv_file, tempmaskfolder='None', labeltype='png', train=True, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.imgs = img_mask['Image'].values.tolist()
        self.masks = img_mask['Mask'].values.tolist()
        self.depths = img_mask['Depth'].values.tolist()
        self.tempmaskfolder = tempmaskfolder
        self.transform = transform
        self.labeltype = labeltype
        self.train = train

    def __getitem__(self, idx):

        imgarr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.imgs[idx])))
        imgarr = imgarr[self.depths[idx], :, :]
        imgarr = imgarr / imgarr.max() * 255.0
        img = Image.fromarray(imgarr)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        maskpath = os.path.join(self.root, self.masks[idx])
        labelgt = False
        if self.train:
            caseid = self.masks[idx].split('/')[-1]
            if 'segmentation' in caseid:
                labelgt = True
                caseid = caseid.split('_')[0]
                pass
            else:
                caseid = self.masks[idx].split('/')[-1]
            maskpath1 = os.path.join(self.root, self.tempmaskfolder, caseid, '{}_depth{}_net1.{}'.
                                     format(caseid, self.depths[idx], self.labeltype))
            maskpath2 = os.path.join(self.root, self.tempmaskfolder, caseid, '{}_depth{}_net2.{}'.
                                     format(caseid, self.depths[idx], self.labeltype))
        else:
            maskpath1 = maskpath
            maskpath2 = maskpath

        if labelgt or self.train == False:
            maskarr = sitk.GetArrayFromImage(sitk.ReadImage(maskpath))
            maskarr = maskarr[self.depths[idx], :, :]
        else:
            maskarr = np.array(Image.open(os.path.join(maskpath, '{}_depth{}.png'.format(caseid, self.depths[idx]))))
        maskarr[maskarr > 0] = 1
        mask = Image.fromarray(maskarr.astype(np.int8))

        if self.train and os.path.exists(maskpath1):
            maskarr1 = np.array(Image.open(maskpath1))
            maskarr1[maskarr1 > 0] = 1
            mask1 = Image.fromarray(maskarr1.astype(np.int8))
        else:
            mask1 = mask.copy()

        if self.train and os.path.exists(maskpath2):
            maskarr2 = np.array(Image.open(maskpath2))
            maskarr2[maskarr2 > 0] = 1
            mask2 = Image.fromarray(maskarr2.astype(np.int8))
        else:
            mask2 = mask.copy()

        augset = {
            'augno': 4,
            'img1': img.copy(),
            'degree1': 0.0,
            'hflip1': 0,
            'img2': img.copy(),
            'degree2': 0.0,
            'hflip2': 0,
            'img3': img.copy(),
            'degree3': 0.0,
            'hflip3': 0,
            'img4': img.copy(),
            'degree4': 0.0,
            'hflip4': 0,
        }
        if self.transform:
            img, augset, mask, mask1, mask2 = self.transform(img, augset, mask, mask1, mask2)

        return img, augset, mask, mask1, mask2

    def __len__(self):
        return len(self.masks)
