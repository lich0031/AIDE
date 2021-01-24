from torch import nn
import torch.nn.functional as F
import torch
import os
import numpy as np

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

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
        loss = 1.0 - (((2. * intersection + self.smooth) / (torch.sum(iflat, dim=1) + torch.sum(tflat, dim=1) + self.smooth)))
        if self.reduction == 'mean':
            diceloss = loss.sum() / inputs.shape[0]
        elif self.reduction == 'sum':
            diceloss = loss.sum()
        elif self.reduction == 'none':
            diceloss = loss
        else:
            print('Wrong')
        return diceloss

class Pixelcoreg_Focalloss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(Pixelcoreg_Focalloss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs1, inputs2, inputs3, targets, forget_rate, kdweight, device):
        batch_size = targets.shape[0]
        # if lossweight == None:
        lossweight = targets.sum().float() / targets.numel() * batch_size
        lossweight = 1

        inputs1_logsoftmax = F.log_softmax(inputs1, dim=1)
        inputs1_softmax = F.softmax(inputs1, dim=1)
        loss_1 = - targets.float() * torch.pow(1 - inputs1_softmax[:, 1, :, :], 2) * inputs1_logsoftmax[:, 1, :, :] - \
                 lossweight * (1 - targets).float() * torch.pow(1 - inputs1_softmax[:, 0, :, :], 2) * inputs1_logsoftmax[:, 0, :, :]
        loss_1 = loss_1.view(batch_size, -1)

        inputs2_logsoftmax = F.log_softmax(inputs2, dim=1)
        inputs2_softmax = F.softmax(inputs2, dim=1)
        loss_2 = - targets.float() * torch.pow(1 - inputs2_softmax[:, 1, :, :], 2) * inputs2_logsoftmax[:, 1, :, :] - \
                 lossweight * (1 - targets).float() * torch.pow(1 - inputs2_softmax[:, 0, :, :], 2) * inputs2_logsoftmax[:, 0, :, :]
        loss_2 = loss_2.view(batch_size, -1)

        inputs3_logsoftmax = F.log_softmax(inputs3, dim=1)
        inputs3_softmax = F.softmax(inputs3, dim=1)
        loss_3 = - targets.float() * torch.pow(1 - inputs3_softmax[:, 1, :, :], 2) * inputs3_logsoftmax[:, 1, :, :] - \
                 lossweight * (1 - targets).float() * torch.pow(1 - inputs3_softmax[:, 0, :, :], 2) * inputs3_logsoftmax[:, 0, :, :]
        loss_3 = loss_3.view(batch_size, -1)
        ind_3_sorted = np.argsort(loss_3.cpu().data).to(device)

        KDL_12 = inputs1_softmax[:, 0, :, :] * torch.log(inputs1_softmax[:, 0, :, :] / inputs2_softmax[:, 0, :, :]) + \
                 inputs1_softmax[:, 1, :, :] * torch.log(inputs1_softmax[:, 1, :, :] / inputs2_softmax[:, 1, :, :])
        KDL_12 = KDL_12.view(batch_size, -1)

        KDL_21 = inputs2_softmax[:, 0, :, :] * torch.log(inputs2_softmax[:, 0, :, :] / inputs1_softmax[:, 0, :, :]) + \
                 inputs2_softmax[:, 1, :, :] * torch.log(inputs2_softmax[:, 1, :, :] / inputs1_softmax[:, 1, :, :])
        KDL_21 = KDL_21.view(batch_size, -1)

        loss = (1-kdweight) * (loss_1 + loss_2 + loss_3) + kdweight * (KDL_12 + KDL_21)

        ind_sorted = np.argsort(loss.cpu().data).to(device)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss_1.shape[1])
        # num_forget = int(forget_rate * loss_1.shape[1])

        ind_update = ind_sorted[:, :num_remember]
        loss_update = loss_3[0, ind_update[0,:]]
        for i in range(batch_size-1):
            loss_update = torch.cat((loss_update, loss_3[i+1, ind_update[i+1,:]]), dim=0)

        loss = loss_update.view(batch_size, -1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            print('Wrong')

        targets_s = targets.view(batch_size, -1)
        targets_sf = targets_s[0, ind_update[0,:]]
        for i in range(batch_size-1):
            targets_sf = torch.cat((targets_sf, targets_s[i+1, ind_update[i+1,:]]), dim=0)

        loss_s = targets_sf.sum() / targets.sum()

        return loss, loss_s

class Pixelcoreg_Focalloss_twomodel(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(Pixelcoreg_Focalloss_twomodel, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs1, inputs2, targets, forget_rate, kdweight, device):
        batch_size = targets.shape[0]
        # if lossweight == None:
        lossweight = targets.sum().float() / targets.numel() * batch_size
        lossweight = 1

        inputs1_logsoftmax = F.log_softmax(inputs1, dim=1)
        inputs1_softmax = F.softmax(inputs1, dim=1)
        loss_1 = - targets.float() * torch.pow(1 - inputs1_softmax[:, 1, :, :], 2) * inputs1_logsoftmax[:, 1, :, :] - \
                 lossweight * (1 - targets).float() * torch.pow(1 - inputs1_softmax[:, 0, :, :], 2) * inputs1_logsoftmax[:, 0, :, :]
        loss_1 = loss_1.view(batch_size, -1)

        inputs2_logsoftmax = F.log_softmax(inputs2, dim=1)
        inputs2_softmax = F.softmax(inputs2, dim=1)
        loss_2 = - targets.float() * torch.pow(1 - inputs2_softmax[:, 1, :, :], 2) * inputs2_logsoftmax[:, 1, :, :] - \
                 lossweight * (1 - targets).float() * torch.pow(1 - inputs2_softmax[:, 0, :, :], 2) * inputs2_logsoftmax[:, 0, :, :]
        loss_2 = loss_2.view(batch_size, -1)

        KDL_12 = inputs1_softmax[:, 0, :, :] * torch.log(inputs1_softmax[:, 0, :, :] / inputs2_softmax[:, 0, :, :]) + \
                 inputs1_softmax[:, 1, :, :] * torch.log(inputs1_softmax[:, 1, :, :] / inputs2_softmax[:, 1, :, :])
        KDL_12 = KDL_12.view(batch_size, -1)

        KDL_21 = inputs2_softmax[:, 0, :, :] * torch.log(inputs2_softmax[:, 0, :, :] / inputs1_softmax[:, 0, :, :]) + \
                 inputs2_softmax[:, 1, :, :] * torch.log(inputs2_softmax[:, 1, :, :] / inputs1_softmax[:, 1, :, :])
        KDL_21 = KDL_21.view(batch_size, -1)

        loss = (1-kdweight) * (loss_1 + loss_2) + kdweight * (KDL_12 + KDL_21)

        ind_sorted = np.argsort(loss.cpu().data).to(device)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss_1.shape[1])
        # num_forget = int(forget_rate * loss_1.shape[1])

        ind_update = ind_sorted[:, :num_remember]
        loss_update = loss[0, ind_update[0,:]]
        for i in range(batch_size-1):
            loss_update = torch.cat((loss_update, loss[i+1, ind_update[i+1,:]]), dim=0)

        loss = loss_update.view(batch_size, -1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            print('Wrong')

        targets_s = targets.view(batch_size, -1)
        targets_sf = targets_s[0, ind_update[0,:]]
        for i in range(batch_size-1):
            targets_sf = torch.cat((targets_sf, targets_s[i+1, ind_update[i+1,:]]), dim=0)

        loss_s = targets_sf.sum() / targets.sum()

        return loss, loss_s