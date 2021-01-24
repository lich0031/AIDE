import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import SimpleITK as sitk

class kidney_seg(Dataset):

    def __init__(self, root, csv_file, tempmaskfolder, maskidentity=1, train=True, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.img = img_mask['Image'].values.tolist()
        self.mask1 = img_mask['Mask1'].values.tolist()
        self.mask2 = img_mask['Mask2'].values.tolist()
        self.mask3 = img_mask['Mask3'].values.tolist()
        self.transform = transform
        self.tempmaskfolder = tempmaskfolder
        self.train = train
        self.maskidentity = maskidentity

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.img[idx])))
        img = img / img.max() * 255.0
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.train:
            maskidentity = {'1': self.mask1, '2': self.mask2, '3': self.mask3}
            maskpath = os.path.join(self.root, maskidentity['{}'.format(int(self.maskidentity))][idx])

            mask = sitk.GetArrayFromImage(sitk.ReadImage(maskpath))

            maskpath1 = os.path.join(self.root, self.tempmaskfolder, maskpath.split('/')[-2],
                                     '{}_net1.nii.gz'.format(maskpath.split('/')[-1].split('.')[0]))
            maskpath2 = os.path.join(self.root, self.tempmaskfolder, maskpath.split('/')[-2],
                                     '{}_net2.nii.gz'.format(maskpath.split('/')[-1].split('.')[0]))
        else:
            mask1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask1[idx])))
            mask2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask2[idx])))
            mask3 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.mask3[idx])))
            mask = (mask1 + mask2 + mask3).astype(np.float) / 3.0

        mask = Image.fromarray(mask[0,:,:])
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 0.5] = 255
        mask_arr[mask_arr <= 0.5] = 0
        mask = Image.fromarray(mask_arr)

        if self.train and os.path.exists(maskpath1):
            mask1 = sitk.GetArrayFromImage(sitk.ReadImage(maskpath1))
            mask1 = Image.fromarray(mask1[0, :, :])
            if mask1.mode != 'L':
                mask1 = mask1.convert('L')
            mask1_arr = np.array(mask1)
            mask1_arr[mask1_arr > 0.5] = 255
            mask1_arr[mask1_arr <= 0.5] = 0
            mask1 = Image.fromarray(mask1_arr)

        else:
            mask1 = mask

        if self.train and os.path.exists(maskpath2):
            mask2 = sitk.GetArrayFromImage(sitk.ReadImage(maskpath2))
            mask2 = Image.fromarray(mask2[0, :, :])
            if mask2.mode != 'L':
                mask2 = mask2.convert('L')
            mask2_arr = np.array(mask2)
            mask2_arr[mask2_arr > 0.5] = 255
            mask2_arr[mask2_arr <= 0.5] = 0
            mask2 = Image.fromarray(mask2_arr)
        else:
            mask2 = mask

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
        return len(self.img)
