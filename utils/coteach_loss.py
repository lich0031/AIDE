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

class CrossEntropyLoss2d2(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d2, self).__init__()
        self.reduction=reduction

    def forward(self, inputs, targets):
        inputs_logsoftmax = F.log_softmax(inputs, dim=1)
        loss = - (1-targets).float()*inputs_logsoftmax[:,0,:,:] - targets.float() * inputs_logsoftmax[:,1,:,:]
        return loss

class Focal_Loss(nn.Module):
    def __init__(self, weight1=1.0, weight2=1.0, beta=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs_softmax = F.softmax(inputs, dim=1)
        inputs1_logsoftmax = F.log_softmax(inputs, dim=1)

        loss = -self.weight1 * torch.pow(inputs_softmax[:,1,:,:], self.beta) * inputs1_logsoftmax[:,0,:,:] * (1-targets).float()\
               -self.weight2 * torch.pow(inputs_softmax[:,0,:,:], self.beta) * inputs1_logsoftmax[:,1,:,:] * targets.float()

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            print('Wrong')
        return loss

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

class CE_Dice_Loss(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super(CE_Dice_Loss, self).__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss2d(reduction=reduction)
        self.dice = Dice_Loss(reduction=reduction)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight * ce_loss + dice_loss

def KLbidirection(inputs1, inputs2):
    inputs1_softmax = F.softmax(inputs1, dim=1)
    inputs2_softmax = F.softmax(inputs2, dim=1)
    KL12 = inputs1_softmax * torch.log(inputs1_softmax / inputs2_softmax)
    KL21 = inputs2_softmax * torch.log(inputs2_softmax / inputs1_softmax)
    KL12 = torch.sum(KL12, dim=1)
    KL21 = torch.sum(KL21, dim=1)
    return KL12 + KL21

class Coteachingloss_dropimage(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super(Coteachingloss_dropimage, self).__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss2d(reduction=reduction)
        self.dice = Dice_Loss(reduction=reduction)

    def forward(self, inputs1, inputs2, targets, forget_rate):
        loss1 = self.weight * torch.mean(self.ce(inputs1, targets), dim=[1,2]) + self.dice(inputs1, targets)
        loss2 = self.weight * torch.mean(self.ce(inputs2, targets), dim=[1,2]) + self.dice(inputs2, targets)
        ind1_sorted = np.argsort(loss1.cpu().data)
        ind2_sorted = np.argsort(loss2.cpu().data)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss1.shape[0])
        num_forget = int(forget_rate * loss1.shape[0])

        ind_1_update = ind1_sorted[:num_remember]
        ind_2_update = ind2_sorted[:num_remember]

        loss1_update = self.weight * torch.mean(self.ce(inputs1[ind_2_update], targets[ind_2_update]), dim=[1,2]) \
                       + self.dice(inputs1[ind_2_update], targets[ind_2_update])
        loss2_update = self.weight * torch.mean(self.ce(inputs2[ind_1_update], targets[ind_1_update]), dim=[1,2]) + \
                       self.dice(inputs2[ind_1_update], targets[ind_1_update])

        return torch.mean(loss1_update, dim=0), torch.mean(loss2_update, dim=0)

class Coteachingloss_weightimage(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super(Coteachingloss_weightimage, self).__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss2d(reduction=reduction)
        self.dice = Dice_Loss(reduction=reduction)

    def forward(self, inputs1, inputs2, targets, forget_rate):
        loss1 = self.weight * torch.mean(self.ce(inputs1, targets), dim=[1,2]) + self.dice(inputs1, targets)
        loss2 = self.weight * torch.mean(self.ce(inputs2, targets), dim=[1,2]) + self.dice(inputs2, targets)
        ind1_sorted = np.argsort(loss1.cpu().data)
        ind2_sorted = np.argsort(loss2.cpu().data)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss1.shape[0])
        num_forget = int(forget_rate * loss1.shape[0])

        ind_1_update = ind1_sorted[:num_remember]
        ind_1_drop = ind1_sorted[num_remember:]
        ind_2_update = ind2_sorted[:num_remember]
        ind_2_drop = ind2_sorted[num_remember:]

        if len(ind_1_drop)>0:
            loss1_update = self.weight * torch.mean(self.ce(inputs1[ind_2_update], targets[ind_2_update]), dim=[1,2]) \
                           + self.dice(inputs1[ind_2_update], targets[ind_2_update]) + 0.1 * \
                           (self.weight * torch.mean(self.ce(inputs1[ind_2_drop], targets[ind_2_drop]), dim=[1,2]) \
                            + self.dice(inputs1[ind_2_drop], targets[ind_2_drop]))
        else:
            loss1_update = self.weight * torch.mean(self.ce(inputs1[ind_2_update], targets[ind_2_update]), dim=[1,2]) \
                           + self.dice(inputs1[ind_2_update], targets[ind_2_update])

        if len(ind_2_drop)>0:
            loss2_update = self.weight * torch.mean(self.ce(inputs2[ind_1_update], targets[ind_1_update]), dim=[1,2]) + \
                           self.dice(inputs2[ind_1_update], targets[ind_1_update]) + 0.1 * \
                           (self.weight * torch.mean(self.ce(inputs2[ind_1_drop], targets[ind_1_drop]), dim=[1, 2]) + \
                            self.dice(inputs2[ind_1_drop], targets[ind_1_drop]))
        else:
            loss2_update = self.weight * torch.mean(self.ce(inputs2[ind_1_update], targets[ind_1_update]), dim=[1,2]) + \
                           self.dice(inputs2[ind_1_update], targets[ind_1_update])

        return torch.mean(loss1_update, dim=0), torch.mean(loss2_update, dim=0)

class Coteachingloss_dropregionce(nn.Module):
    def __init__(self, scale=0.5, reduction='none'):
        super(Coteachingloss_dropregionce, self).__init__()
        self.scale = scale
        self.ce = CrossEntropyLoss2d(reduction=reduction)
        # self.dice = Dice_Loss(reduction=reduction)

    def forward(self, inputs1, inputs2, targets, forget_rate):
        total_w, total_h = inputs1.shape[2], inputs1.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        kernel_w, kernel_h = int(total_w/patch_w), int(total_h/patch_h)
        maxpool = nn.MaxPool2d(kernel_size=(kernel_w, kernel_h), stride=(kernel_w, kernel_h), padding=0, ceil_mode=True)
        inputs1_pool = maxpool(inputs1)
        inputs2_pool = maxpool(inputs2)
        targets_pool = maxpool(targets.float()).long()
        loss1 = self.ce(inputs1_pool, targets_pool).view(inputs1_pool.shape[0],-1)
        loss2 = self.ce(inputs2_pool, targets_pool).view(inputs2_pool.shape[0],-1)
        ind1_sorted = np.argsort(loss1.cpu().data)
        ind2_sorted = np.argsort(loss2.cpu().data)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss1.shape[1])
        num_forget = int(forget_rate * loss1.shape[1])

        ind_1_update = ind1_sorted[:,:num_remember]
        ind_2_update = ind2_sorted[:,:num_remember]

        loss1_update = loss1[0,ind_2_update[0,:]]
        loss2_update = loss2[0,ind_1_update[0,:]]

        for i in range(inputs1.shape[0]-1):
            loss1_update = torch.cat([loss1_update, loss1[i+1, ind_2_update[i+1, :]]], dim=0)
            loss2_update = torch.cat([loss2_update, loss2[i + 1, ind_1_update[i + 1, :]]], dim=0)
        return torch.mean(loss1_update), torch.mean(loss2_update)

class Coteachingloss_dropimagedroppixel(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super(Coteachingloss_dropimagedroppixel, self).__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss2d(reduction=reduction)
        self.dice = Dice_Loss(reduction=reduction)

    def forward(self, inputs1, inputs2, targets, forget_rate):
        loss1 = self.weight * torch.mean(self.ce(inputs1, targets), dim=[1,2]) + self.dice(inputs1, targets)
        loss2 = self.weight * torch.mean(self.ce(inputs2, targets), dim=[1,2]) + self.dice(inputs2, targets)
        ind1_sorted = np.argsort(loss1.cpu().data)
        ind2_sorted = np.argsort(loss2.cpu().data)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss1.shape[0])
        num_forget = int(forget_rate * loss1.shape[0])

        ind_1_update = ind1_sorted[:num_remember]
        ind_2_update = ind2_sorted[:num_remember]

        loss1_update = self.weight * torch.mean(self.ce(inputs1[ind_2_update], targets[ind_2_update]), dim=[1,2]) \
                       + self.dice(inputs1[ind_2_update], targets[ind_2_update])
        loss2_update = self.weight * torch.mean(self.ce(inputs2[ind_1_update], targets[ind_1_update]), dim=[1,2]) + \
                       self.dice(inputs2[ind_1_update], targets[ind_1_update])

        ind_1_drop = ind1_sorted[num_remember:]
        ind_2_drop = ind2_sorted[num_remember:]
        if len(ind_1_drop) > 0:
            inputs1_drop = inputs1[ind_2_drop]
            inputs2_drop_according = inputs2[ind_2_drop]
            targets1_drop = targets[ind_2_drop]
            klloss1 = KLbidirection(inputs1_drop, inputs2_drop_according)
            inputs1_dropce = self.ce(inputs1_drop, targets1_drop)
            loss1_drop = (klloss1 + inputs1_dropce).view(-1) * targets1_drop.view(-1).float()
            loss1_drop_fore = loss1_drop[loss1_drop>0]
            ind1drop_sorted = np.argsort(loss1_drop_fore.cpu().data)
            num_remember2 = int(remember_rate * len(ind1drop_sorted))
            ind1drop_update = ind1drop_sorted[:num_remember2]
            loss1drop_update = torch.mean(loss1_drop_fore[ind1drop_update])
        else:
            loss1drop_update = 0.0

        if len(ind_2_drop) > 0:
            inputs2_drop = inputs2[ind_1_drop]
            inputs1_drop_according = inputs1[ind_1_drop]
            targets2_drop = targets[ind_1_drop]
            klloss2 = KLbidirection(inputs1_drop_according, inputs2_drop)
            inputs2_dropce = self.ce(inputs2_drop, targets2_drop)
            loss2_drop = (klloss2 + inputs2_dropce).view(-1) * targets2_drop.view(-1).float()
            loss2_drop_fore = loss2_drop[loss2_drop>0]
            ind2drop_sorted = np.argsort(loss2_drop_fore.cpu().data)
            ind2drop_update = ind2drop_sorted[:num_remember2]
            loss2drop_update = torch.mean(loss2_drop_fore[ind2drop_update])
        else:
            loss2drop_update = 0.0

        return torch.mean(loss1_update, dim=0) + 0.25 * loss1drop_update, torch.mean(loss2_update, dim=0) + 0.25 * loss2drop_update


if __name__=='__main__':
    a = torch.rand([4,2,8,8])
    b = torch.rand([4,2,8,8])
    c = torch.ones([4,8,8]).long()
    # celoss = Dice_Loss()
    # celoss(a,c)
    criterion1 = Coteachingloss_dropimagedroppixel(reduction='none')
    criterion1(a,b,c,0.2)
    # criterion2(a,c)
    # criterion3(a,c)