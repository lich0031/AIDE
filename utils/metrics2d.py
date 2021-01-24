import torch.nn.functional as F
import torch
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt

def Dice_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs = inputs[:, 1, :, :]
    inputs[inputs >= threshold] = 1
    inputs[inputs < threshold] = 0
    dice = 0.
    img_count = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        if tflat.sum() == 0:
            if iflat.sum() == 0:
                dice_single = torch.tensor(1.0)
            else:
                dice_single = torch.tensor(0.0)
                img_count += 1
        else:
            dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
            img_count += 1
        dice += dice_single
    return dice

def Dice_fn_Nozero(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs = inputs[:, 1, :, :]
    inputs[inputs >= threshold] = 1
    inputs[inputs < threshold] = 0
    dice = 0.
    img_count = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        if tflat.sum() == 0:
            if iflat.sum() == 0:
                dice_single = torch.tensor(1.0)
            else:
                dice_single = torch.tensor(0.0)
                img_count += 1
        else:
            dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
            img_count += 1
        dice += dice_single
    return dice.item(), img_count

def TP_TN_FP_FN(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        TP = (iflat * tflat).sum()
        TN = ((1 - iflat) * (1 - tflat)).sum()
        FP = (iflat * (1 - tflat)).sum()
        FN = ((1 - iflat) * tflat).sum()
    return TP, TN, FP, FN

def IoU_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs_obj = inputs[:, 1, :, :]
    inputs_obj[inputs_obj >= threshold] = 1
    inputs_obj[inputs_obj < threshold] = 0
    IoU = 0.
    for input_, target_ in zip(inputs_obj, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        IoU_single = intersection / (iflat.sum() + tflat.sum() - intersection)
        IoU += IoU_single
    return IoU

def MulticlassAccuracy_fn(inputs, targets, mode='eval'):
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.unsqueeze(inputs, dim=1)
    inputs = inputs.numpy()
    N = targets.shape[0]
    w, h = targets.shape[2:]

    label_values = [[0], [1], [2], [3], [4]]
    inputs = one_hot_result(inputs, label_values).astype(np.float)
    categories = targets.shape[1]

    correct_pred = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.reshape(categories, -1)
        tflat = target_.reshape(categories, -1)
        intersection = iflat * tflat
        correct_pred += intersection.sum()
    if 'train3_multidomainl_normalcl' in mode:
        accuracy = correct_pred.astype(np.float) / float(w) / float(h)
    else:
        accuracy = correct_pred.astype(np.float) / float(N)
    return accuracy

def MulticlassDice_fn(inputs, targets, mode='eval'):
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.unsqueeze(inputs, dim=1)
    inputs = inputs.numpy()
    N = targets.shape[0]
    categories = targets.shape[1]

    label_values = np.arange(categories)
    inputs = one_hot_result(inputs, label_values).astype(np.float)

    dice = np.zeros(categories)
    for input_, target_ in zip(inputs, targets):
        iflat = input_.reshape(categories, -1)
        tflat = target_.reshape(categories, -1)
        intersection = iflat * tflat
        intersection = 2 * np.sum(intersection, axis=1)
        union = np.sum(iflat, axis=1) + np.sum(tflat, axis=1)
        dice_image = intersection / union
        dice_image[np.isnan(dice_image) * (union==0)] = 1.0
        dice_image[np.isnan(dice_image) * (union!=0)] = 0.0
        dice += dice_image
    if 'train3_multidomainl_normalcl' in mode:
        dice = dice[1:].sum() / (categories-1)
    else:
        dice = dice / float(N)
    return dice

def MulticlassIoU_fn(inputs, targets, mode='eval'):
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.unsqueeze(inputs, dim=1)
    inputs = inputs.numpy()
    N = targets.shape[0]
    categories = targets.shape[1]

    label_values = np.arange(categories)
    inputs = one_hot_result(inputs, label_values).astype(np.float)

    iou = np.zeros(categories)
    for input_, target_ in zip(inputs, targets):
        iflat = input_.reshape(categories, -1)
        tflat = target_.reshape(categories, -1)
        intersection = iflat * tflat
        intersection = np.sum(intersection, axis=1)
        union = np.sum(iflat, axis=1) + np.sum(tflat, axis=1)
        iou_image = intersection / (union - intersection)
        iou_image[np.isnan(iou_image) * (union==0)] = 1.0
        iou_image[np.isnan(iou_image) * (union!=0)] = 0.0
        iou += iou_image
    if 'train3_multidomainl_normalcl' in mode:
        iou = iou.sum() / float(N) / categories
    else:
        iou = iou / float(N)
    return iou

def MulticlassTP_TN_FP_FN(inputs, targets, mode='eval'):
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.unsqueeze(inputs, dim=1)
    inputs = inputs.numpy()
    N = targets.shape[0]
    categories = targets.shape[1]

    label_values = np.arange(categories)
    inputs = one_hot_result(inputs, label_values).astype(np.float)

    TP = np.zeros(categories)
    TN = np.zeros(categories)
    FP = np.zeros(categories)
    FN = np.zeros(categories)
    for input_, target_ in zip(inputs, targets):
        iflat = input_.reshape(categories, -1)
        tflat = target_.reshape(categories, -1)
        TP += np.sum(iflat * tflat, axis=1)
        TN += np.sum((1 - iflat) * (1 - tflat), axis=1)
        FP += np.sum(iflat * (1 - tflat), axis=1)
        FN += np.sum((1 - iflat) * tflat, axis=1)
    TP = TP / float(N)
    TN = TN / float(N)
    FP = FP / float(N)
    FN = FN / float(N)
    return TP, TN, FP, FN

def one_hot_result(label, label_values=[[0], [1], [2], [3], [4]]):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=1)
    return semantic_map