from torch import nn
import torch.nn.functional as F
import torch

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        if len(targets.shape)>3:
            targets = torch.argmax(targets.float(), dim=1)
        return self.crossentropy_loss(inputs, targets)

def diceloss2d(inputs, targets, smooth=1.0, reduction='mean'):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    diceloss = []

    iflat = inputs_obj.view(inputs.shape[0], -1).float()
    tflat = targets.view(targets.shape[0], -1).float()
    intersection = torch.sum(iflat*tflat, dim=1)
    loss = 1.0 - (((2. * intersection + smooth) / (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + smooth)))

    if reduction == 'mean':
        diceloss = loss.sum() / inputs.shape[0]
    elif reduction == 'sum':
        diceloss = loss.sum()
    elif reduction == 'none':
        diceloss = loss
    else:
        print('Wrong')
    return diceloss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        N = target.size(0)
        if len(input.shape) > 3:
            input = F.softmax(input, dim=1)
            iflat = input[:,1,:,:].view(N, -1).float()
        else:
            iflat = input.view(N, -1).float()
        tflat = target.view(N, -1).float()
        intersection = iflat*tflat
        loss = 1.0 - (2. * intersection.sum(1) + self.smooth) / \
               (iflat.sum(1) + tflat.sum(1) + self.smooth)
        if self.reduction == 'mean':
            diceloss = loss.sum() / N
        elif self.reduction == 'sum':
            diceloss = loss.sum()
        elif self.reduction == 'none':
            diceloss = loss
        else:
            print('Wrong')
        return diceloss

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        inputs_obj = inputs[:, 1, :, :]
        iflat = inputs_obj.view(inputs.shape[0], -1).float()
        tflat = targets.view(targets.shape[0], -1).float()
        intersection = torch.sum(iflat*tflat, dim=1)
        loss = 1.0 - (((2. * intersection + self.smooth) /
                       (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + self.smooth)))
        if self.reduction == 'mean':
            diceloss = loss.sum() / inputs.shape[0]
        elif self.reduction == 'sum':
            diceloss = loss.sum()
        elif self.reduction == 'none':
            diceloss = loss
        else:
            print('Wrong')
        return diceloss

class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0, reduction='mean'):
        super(MulticlassDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction
        self.dice = DiceLoss(smooth=self.smooth, reduction=self.reduction)

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        totalLoss = 0
        if len(target.shape)>3:
            C = target.shape[1]
            for i in range(C):
                diceLoss = self.dice(input[:,i], target[:,i])
                if self.weight is not None:
                    diceLoss *= self.weight[i]
                totalLoss += diceLoss
        else:
            totalLoss = self.dice(input[:,1], target)
        return totalLoss

class MulticlassMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MulticlassMSELoss, self).__init__()
        self.reduction = reduction
        self.mseloss = nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        return self.mseloss(input, target)

class CEMDiceLoss(nn.Module):
    def __init__(self, cediceweight=None, ceclassweight=None, diceclassweight=None, reduction='mean'):
        super(CEMDiceLoss, self).__init__()
        self.cediceweight = cediceweight
        self.ceclassweight = ceclassweight
        self.diceclassweight = diceclassweight
        self.ce = CrossEntropyLoss2d(weight=ceclassweight, reduction=reduction)
        self.multidice = MulticlassDiceLoss(weight=diceclassweight, reduction=reduction)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.multidice(inputs, targets)
        if self.cediceweight is not None:
            loss = ce_loss * self.cediceweight[0] + dice_loss * self.cediceweight[1]
        else:
            loss = ce_loss + dice_loss
        return loss

class CEMDiceLossImage(nn.Module):
    def __init__(self, cediceweight=None, ceclassweight=None, diceclassweight=None, reduction='mean'):
        super(CEMDiceLossImage, self).__init__()
        self.cediceweight = cediceweight
        self.ceclassweight = ceclassweight
        self.diceclassweight = diceclassweight
        self.ce = CrossEntropyLoss2d(weight=ceclassweight, reduction='none')
        self.multidice = MulticlassDiceLoss(weight=diceclassweight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        ce_loss = ce_loss.mean(dim=[1,2])
        dice_loss = self.multidice(inputs, targets)
        if self.cediceweight is not None:
            loss = ce_loss * self.cediceweight[0] + dice_loss * self.cediceweight[1]
        else:
            loss = ce_loss + dice_loss
        return loss

class CEDiceLoss(nn.Module):
    def __init__(self, cediceweight=None, classweight=None, reduction='mean'):
        super(CEDiceLoss, self).__init__()
        self.cediceweight = cediceweight
        self.classweight = classweight
        self.ce = CrossEntropyLoss2d(weight=classweight, reduction=reduction)
        self.dice = DiceLoss(weight=classweight, reduction=reduction)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        if self.cediceweight is not None:
            loss = ce_loss * self.cediceweight[0] + dice_loss * self.cediceweight[1]
        else:
            loss = ce_loss + dice_loss
        return loss
