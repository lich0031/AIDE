import random

from PIL import Image
import numpy as np
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, augset, mask):
        for t in self.transforms:
            img, augset, mask = t(img, augset, mask)
        return img, augset, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, augset, mask):
        if random.random() < 0.5:
            augset['img1'] = augset['img1'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip1'] = 1
        if random.random() < 0.5:
            augset['img2'] = augset['img2'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip2'] = 1
        if random.random() < 0.5:
            augset['img3'] = augset['img3'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip3'] = 1
        if random.random() < 0.5:
            augset['img4'] = augset['img4'].transpose(Image.FLIP_LEFT_RIGHT)
            augset['hflip4'] = 1
        return img, augset, mask

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, augset, mask):
        assert len(self.size) == 2
        ow, oh = self.size
        augset['img1'] = augset['img1'].resize((ow, oh), Image.BILINEAR)
        augset['img2'] = augset['img2'].resize((ow, oh), Image.BILINEAR)
        augset['img3'] = augset['img3'].resize((ow, oh), Image.BILINEAR)
        augset['img4'] = augset['img4'].resize((ow, oh), Image.BILINEAR)

        return img.resize((ow, oh), Image.BILINEAR), \
               augset, mask.resize((ow, oh), Image.NEAREST)

class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img, augset, mask):

        augset['img1'] = Image.fromarray(augset['img1'])
        augset['img2'] = Image.fromarray(augset['img2'])
        augset['img3'] = Image.fromarray(augset['img3'])
        augset['img4'] = Image.fromarray(augset['img4'])

        return Image.fromarray(img), augset, Image.fromarray(mask)

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, augset, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['img1'] = augset['img1'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree1'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['img2'] = augset['img2'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree2'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['img3'] = augset['img3'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree3'] = rotate_degree

        rotate_degree = random.random() * 2 * self.degree - self.degree
        augset['img4'] = augset['img4'].rotate(rotate_degree, Image.BILINEAR)
        augset['degree4'] = rotate_degree

        return img, augset, mask

class ToTensor(object):
    """ puts channels in front and convert to float, except if mode palette
    """
    def __init__(self):
        pass

    def __call__(self, img, augset, mask):
        img = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 255.0
        mask = np.array(mask)
        num_classes = len(np.unique(mask))
        mask = torch.from_numpy(mask)
        mask = ((num_classes -1) * mask.float() / 255.0).long()

        augset['img1'] = torch.from_numpy(np.array(augset['img1']).transpose(2,0,1)).float() / 255.0
        augset['img2'] = torch.from_numpy(np.array(augset['img2']).transpose(2, 0, 1)).float() / 255.0
        augset['img3'] = torch.from_numpy(np.array(augset['img3']).transpose(2, 0, 1)).float() / 255.0
        augset['img4'] = torch.from_numpy(np.array(augset['img4']).transpose(2, 0, 1)).float() / 255.0

        return img, augset, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, augset, mask):

        if self.mean is None:
            imgmean = img.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            imgstd = img.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
        else:
            imgmean = torch.FloatTensor(self.mean).unsqueeze(1).unsqueeze(2)
            imgstd = torch.FloatTensor(self.std).unsqueeze(1).unsqueeze(2)

        img = img.sub(imgmean).div(imgstd)

        augset['img1'] = augset['img1'].sub(imgmean).div(imgstd)
        augset['img2'] = augset['img2'].sub(imgmean).div(imgstd)
        augset['img3'] = augset['img3'].sub(imgmean).div(imgstd)
        augset['img4'] = augset['img4'].sub(imgmean).div(imgstd)

        return img, augset, mask