import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import SimpleITK as sitk

class breast_seg(Dataset):

    def __init__(self, root, csv_file, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.imgs = img_mask['Image'].values.tolist()
        self.masks = img_mask['Mask'].values.tolist()
        self.depths = img_mask['Depth'].values.tolist()
        self.transform = transform

    def __getitem__(self, idx):
        imgarr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.imgs[idx])))
        imgarr = imgarr[self.depths[idx], :, :]
        imgarr = imgarr / imgarr.max() * 255.0
        img = Image.fromarray(imgarr)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        maskarr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.masks[idx])))
        if maskarr.shape[0] < self.depths[idx]:
            print(self.masks[idx])
        maskarr = maskarr[self.depths[idx], :, :]
        maskarr[maskarr > 0] = 1
        mask = Image.fromarray(maskarr.astype(np.int8))

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
            img, augset, mask = self.transform(img, augset, mask)
        return img, augset, mask

    def __len__(self):
        return len(self.masks)