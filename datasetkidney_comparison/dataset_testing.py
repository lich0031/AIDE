import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import SimpleITK as sitk

class kidney_seg_testing(Dataset):

    def __init__(self, root, csv_file):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.img = img_mask['Image'].values.tolist()

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root, self.img[idx])))
        img = img[0,:,:] / img.max() * 255.0
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.img)
