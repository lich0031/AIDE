import random

from PIL import Image, ImageOps
import numpy as np
import torch
import numbers
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

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, augset, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, augset, mask
        if w < tw or h < th:
            augset['img1'] = augset['img1'].resize((tw, th), Image.BILINEAR)

            augset['img2'] = augset['img2'].resize((tw, th), Image.BILINEAR)

            augset['img3'] = augset['img3'].resize((tw, th), Image.BILINEAR)

            augset['img4'] = augset['img4'].resize((tw, th), Image.BILINEAR)

            return img.resize((tw, th), Image.BILINEAR), augset, mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        augset['img1'] = augset['img1'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img2'] = augset['img2'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img3'] = augset['img3'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img4'] = augset['img4'].crop((x1, y1, x1 + tw, y1 + th))
        return img.crop((x1, y1, x1 + tw, y1 + th)), augset, mask.crop((x1, y1, x1 + tw, y1 + th))

class CenterCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, augset, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, augset, mask
        if w < tw or h < th:
            augset['img1'] = augset['img1'].resize((tw, th), Image.BILINEAR)

            augset['img2'] = augset['img2'].resize((tw, th), Image.BILINEAR)

            augset['img3'] = augset['img3'].resize((tw, th), Image.BILINEAR)

            augset['img4'] = augset['img4'].resize((tw, th), Image.BILINEAR)
            return img.resize((tw, th), Image.BILINEAR), augset, mask.resize((tw, th), Image.NEAREST)

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        augset['img1'] = augset['img1'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img2'] = augset['img2'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img3'] = augset['img3'].crop((x1, y1, x1 + tw, y1 + th))

        augset['img4'] = augset['img4'].crop((x1, y1, x1 + tw, y1 + th))
        return img.crop((x1, y1, x1 + tw, y1 + th)), augset, mask.crop((x1, y1, x1 + tw, y1 + th))

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
        mask = torch.from_numpy(mask).long()

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