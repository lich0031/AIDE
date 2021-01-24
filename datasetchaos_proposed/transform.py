import random

from PIL import Image
import numpy as np
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, augset, mask, mask1, mask2):
        for t in self.transforms:
            img1, img2, augset, mask, mask1, mask2 = t(img1, img2, augset, mask, mask1, mask2)
        return img1, img2, augset, mask, mask1, mask2

class RandomHorizontallyFlip(object):
    def __call__(self, img1, img2, augset, mask, mask1, mask2):
        if random.random() < 0.5:
            augset['imgmodal11'] = augset['imgmodal11'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['imgmodal21'] = augset['imgmodal21'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip1'] = 1
        if random.random() < 0.5:
            augset['imgmodal12'] = augset['imgmodal12'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['imgmodal22'] = augset['imgmodal22'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip2'] = 1
        if random.random() < 0.5:
            augset['imgmodal13'] = augset['imgmodal13'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['imgmodal23'] = augset['imgmodal23'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip3'] = 1
        if random.random() < 0.5:
            augset['imgmodal14'] = augset['imgmodal14'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['imgmodal24'] = augset['imgmodal24'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip4'] = 1
        return img1, img2, augset, mask, mask1, mask2

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2, augset, mask, mask1, mask2):
        assert len(self.size) == 2
        ow, oh = self.size

        augset['imgmodal11'] = augset['imgmodal11'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal21'] = augset['imgmodal21'].resize((ow, oh), Image.BILINEAR)

        augset['imgmodal12'] = augset['imgmodal12'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal22'] = augset['imgmodal22'].resize((ow, oh), Image.BILINEAR)

        augset['imgmodal13'] = augset['imgmodal13'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal23'] = augset['imgmodal23'].resize((ow, oh), Image.BILINEAR)

        augset['imgmodal14'] = augset['imgmodal14'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal24'] = augset['imgmodal24'].resize((ow, oh), Image.BILINEAR)

        return img1.resize((ow, oh), Image.BILINEAR), img2.resize((ow, oh), Image.BILINEAR), \
               augset, mask.resize((ow, oh), Image.NEAREST), mask1.resize((ow, oh), Image.NEAREST), \
               mask2.resize((ow, oh), Image.NEAREST)

class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img1, img2, augset, mask, mask1, mask2):

        augset['imgmodal11'] = Image.fromarray(augset['imgmodal11'])
        augset['imgmodal21'] = Image.fromarray(augset['imgmodal21'])

        augset['imgmodal12'] = Image.fromarray(augset['imgmodal12'])
        augset['imgmodal22'] = Image.fromarray(augset['imgmodal22'])

        augset['imgmodal13'] = Image.fromarray(augset['imgmodal13'])
        augset['imgmodal23'] = Image.fromarray(augset['imgmodal23'])

        augset['imgmodal14'] = Image.fromarray(augset['imgmodal14'])
        augset['imgmodal24'] = Image.fromarray(augset['imgmodal24'])

        return Image.fromarray(img1), Image.fromarray(img2), augset, \
               Image.fromarray(mask), Image.fromarray(mask1), Image.fromarray(mask2)

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img1, img2, augset, mask, mask1, mask2):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['imgmodal11'] = augset['imgmodal11'].rotate(rotate_degree, Image.BILINEAR)
        augset['imgmodal21'] = augset['imgmodal21'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree1'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['imgmodal12'] = augset['imgmodal12'].rotate(rotate_degree, Image.BILINEAR)
        augset['imgmodal22'] = augset['imgmodal22'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree2'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['imgmodal13'] = augset['imgmodal13'].rotate(rotate_degree, Image.BILINEAR)
        augset['imgmodal23'] = augset['imgmodal23'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree3'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['imgmodal14'] = augset['imgmodal14'].rotate(rotate_degree, Image.BILINEAR)
        augset['imgmodal24'] = augset['imgmodal24'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree4'] = rotate_degree

        return img1, img2, augset, mask, mask1, mask2

class ToTensor(object):
    """ puts channels in front and convert to float, except if mode palette
    """
    def __init__(self):
        pass

    def __call__(self, img1, img2, augset, mask, mask1, mask2):
        img1 = torch.from_numpy(np.array(img1).transpose(2,0,1)).float() / 255.0
        img2 = torch.from_numpy(np.array(img2).transpose(2,0,1)).float() / 255.0
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask1 = np.array(mask1)
        mask1 = torch.from_numpy(mask1)
        mask2 = np.array(mask2)
        mask2 = torch.from_numpy(mask2)

        augset['imgmodal11'] = torch.from_numpy(np.array(augset['imgmodal11']).transpose(2,0,1)).float() / 255.0
        augset['imgmodal21'] = torch.from_numpy(np.array(augset['imgmodal21']).transpose(2,0,1)).float() / 255.0

        augset['imgmodal12'] = torch.from_numpy(np.array(augset['imgmodal12']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal22'] = torch.from_numpy(np.array(augset['imgmodal22']).transpose(2, 0, 1)).float() / 255.0

        augset['imgmodal13'] = torch.from_numpy(np.array(augset['imgmodal13']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal23'] = torch.from_numpy(np.array(augset['imgmodal23']).transpose(2, 0, 1)).float() / 255.0

        augset['imgmodal14'] = torch.from_numpy(np.array(augset['imgmodal14']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal24'] = torch.from_numpy(np.array(augset['imgmodal24']).transpose(2, 0, 1)).float() / 255.0

        return img1, img2, augset, mask, mask1, mask2


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img1, img2, augset, mask, mask1, mask2):

        if self.mean is None:
            img1mean = img1.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img1std = img1.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img2mean = img2.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img2std = img2.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
        else:
            img1mean = torch.FloatTensor(self.mean).unsqueeze(1).unsqueeze(2)
            img2mean = img1mean
            img1std = torch.FloatTensor(self.std).unsqueeze(1).unsqueeze(2)
            img2std = img1std

        img1 = img1.sub(img1mean).div(img1std)
        img2 = img2.sub(img2mean).div(img2std)

        augset['imgmodal11'] = augset['imgmodal11'].sub(img1mean).div(img1std)
        augset['imgmodal21'] = augset['imgmodal21'].sub(img2mean).div(img2std)

        augset['imgmodal12'] = augset['imgmodal12'].sub(img1mean).div(img1std)
        augset['imgmodal22'] = augset['imgmodal22'].sub(img2mean).div(img2std)

        augset['imgmodal13'] = augset['imgmodal13'].sub(img1mean).div(img1std)
        augset['imgmodal23'] = augset['imgmodal23'].sub(img2mean).div(img2std)

        augset['imgmodal14'] = augset['imgmodal14'].sub(img1mean).div(img1std)
        augset['imgmodal24'] = augset['imgmodal24'].sub(img2mean).div(img2std)

        return img1, img2, augset, mask, mask1, mask2