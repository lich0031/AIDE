import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import SimpleITK as sitk

class kidney_seg(Dataset):

    def __init__(self, root, csv_file, maskidentity=1, train=True, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.img = img_mask['Image'].values.tolist()
        self.mask1 = img_mask['Mask1'].values.tolist()
        self.mask2 = img_mask['Mask2'].values.tolist()
        self.mask3 = img_mask['Mask3'].values.tolist()
        self.train = train
        self.transform = transform
        self.maskidentity = maskidentity

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.img[idx])))
        img = img / img.max() * 255.0
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.train:
            maskidentitys = {'1': self.mask1, '2': self.mask2, '3': self.mask3}
            mask = sitk.GetArrayFromImage(sitk.ReadImage
                                          (os.path.join(self.root, maskidentitys['{}'.format(int(self.maskidentity))][idx])))

            mask = Image.fromarray(mask)
        else:
            mask1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask1[idx])))
            mask2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask2[idx])))
            mask3 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask3[idx])))
            mask = (mask1 + mask2 + mask3).astype(np.float) / 3.0
            mask = Image.fromarray(mask)

        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 0.5] = 255
        mask_arr[mask_arr <= 0.5] = 0
        mask = Image.fromarray(mask_arr)

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
        return len(self.img)
