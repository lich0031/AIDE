import sys

sys.path.extend(['../../', '../'])
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
from datasetchaos_comparison import chaos_seg, Compose, Resize, ToTensor, Normalize
from models_twomodalinputs import fuseunet
from skimage import measure

palette = [[0], [63], [126], [189], [252]]

def build_model(model_name, num_classes):
    if model_name == 'fuseunet':
        net = fuseunet(num_classes=num_classes)
    else:
        print('wait a minute')
    return net

def plotresult(img, mask, output, savename):
    imgshow = img.copy()
    multiplier = mask[:,:,0]
    multiplier = np.expand_dims(multiplier, axis=2)
    multiplier = np.concatenate([multiplier, multiplier, multiplier], axis=-1)
    imgmaskblend = imgshow * multiplier
    mask = mask.squeeze()
    output = torch.argmax(output, dim=1)
    output = np.expand_dims(output.squeeze().numpy(), axis=2)
    output = one_hot_mask(output, palette=[[0], [1], [2], [3], [4]])
    multiplier = output[:,:,0]
    multiplier = np.expand_dims(multiplier, axis=2)
    multiplier = np.concatenate([multiplier, multiplier, multiplier], axis=-1)
    imgoutputblend = imgshow * multiplier
    maskshow = combineimg(mask)
    outputshow = combineimg(output)
    maskimgshow = imgmaskblend  + maskshow
    outputimgshow = imgoutputblend  + outputshow

    fig, axes = plt.subplots(1, 5, sharey=True)
    ax0, ax1, ax2, ax3, ax4 = axes.ravel()
    ax0.imshow(imgshow)
    ax0.set_title("image")
    ax1.imshow(maskshow)
    ax1.set_title("mask")
    ax2.imshow(outputshow)
    ax2.set_title("output")
    ax3.imshow(maskimgshow)
    ax3.set_title("image-mask")
    ax4.imshow(outputimgshow)
    ax4.set_title("image-output")

    for ax in axes.ravel():
        ax.axis('off')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(savename, format='png', transparent=True, dpi=300, pad_inches=0)
    plt.close()

def generatemask(output, category, savedir):
    output = F.softmax(output, dim=1)
    output = output[0,1,:,:].squeeze()
    output = output.cpu()
    output = Image.fromarray(output.numpy().astype(np.float32), 'F')
    output.save(savedir)

def keep_largeste_connected_components(mask):
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    blobs = measure.label(mask, connectivity=1)         #connectivity 1: 4 neighbours 2: 8 neighbours
    props = measure.regionprops(blobs)
    area = [ele.area for ele in props]
    if mask.max() > 0:
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs==largest_blob_label] = 1
    return out_img

def one_hot_mask(label, palette):
    semantic_map = []
    for color in palette:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map.astype(np.uint8))
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def reversemask(mask, palette=[[0], [63], [126], [189], [252]]):
    mask[mask==0] = palette[0][0]
    mask[mask==1] = palette[1][0]
    mask[mask==2] = palette[2][0]
    mask[mask==3] = palette[3][0]
    mask[mask==4] = palette[4][0]
    return mask

def combineimg(img):
    channel = img.shape[2]
    img = img.astype(np.int8)
    assert channel == 5
    d1 = img[:,:,1] * 255
    d2 = img[:,:,2] * 255
    d3 = img[:,:,3] * 255
    d4 = img[:,:,4] * 255
    imgshow = np.zeros(img.shape[0:2])
    imgshow = np.expand_dims(imgshow, axis=2)
    imgshow = np.concatenate([imgshow, imgshow, imgshow], axis=-1)
    imgshow[:,:,0] = d1 + d4
    imgshow[:,:,1] = d2
    imgshow[:,:,2] = d3 + d4
    return imgshow.astype(np.int)

def Dice3d_fn(inputs, targets):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = iflat * tflat
    intersection = 2 * np.sum(intersection)
    union = np.sum(iflat) + np.sum(tflat)
    dice_image = intersection / union
    return dice_image

def IoU3d_fn(inputs, targets):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = iflat * tflat
    intersection = np.sum(intersection)
    union = np.sum(iflat) + np.sum(tflat)
    iou_image = intersection / (union - intersection)
    return iou_image

def TP_TN_FP_FN3d(inputs, targets):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    TP = np.sum(iflat * tflat)
    TN = np.sum((1 - iflat) * (1 - tflat))
    FP = np.sum(iflat * (1 - tflat))
    FN = np.sum((1 - iflat) * tflat)
    return TP, TN, FP, FN

def eval(modelname, num_classes, device, data_root, casecsv_file,
         imgcsv_file, checkpoint, result_dir, result_csv_name):

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    generatedmasksaveroot = os.path.join(result_dir, 'generated_masks')
    if not os.path.exists(generatedmasksaveroot):
        os.mkdir(generatedmasksaveroot)

    # net
    net = build_model(modelname, num_classes)
    net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()

    # data
    test_cases = pd.read_csv(casecsv_file)['patient_case'].tolist()
    test_dataset = chaos_seg(root=data_root, csv_file=imgcsv_file)
    test_inphaseimgs = test_dataset.t1inphase
    test_outphaseimgs = test_dataset.t1outphase
    test_masks = test_dataset.masks

    # result
    all_result = []
    for caseidx in test_cases:
        result_save_folder = os.path.join(generatedmasksaveroot, str(caseidx))
        if not os.path.exists(result_save_folder):
            os.mkdir(result_save_folder)
        caseinphaseimg = [file for file in test_inphaseimgs if int(file.split('/')[0])==caseidx]
        caseinphaseimg.sort()
        caseoutphaseimg = [file for file in test_outphaseimgs if int(file.split('/')[0])==caseidx]
        caseoutphaseimg.sort()
        casemask = [file for file in test_masks if int(file.split('/')[0])==caseidx]
        casemask.sort()
        generatedtarget = []
        target = []
        imgsavename = []
        voxelspacing = [0, 0, 0]
        for i in tqdm(range(len(caseinphaseimg)), total=len(caseinphaseimg)):
            assert caseinphaseimg[i].split('/')[-1].split('.')[0] == \
                   casemask[i].split('/')[-1].split('.')[0]
            assert caseinphaseimg[i].split('/')[-1].split('-')[1] == \
                   caseoutphaseimg[i].split('/')[-1].split('-')[1]
            assert int(caseinphaseimg[i].split('/')[-1].split('-')[-1].split('.')[0]) == \
                   int(caseoutphaseimg[i].split('/')[-1].split('-')[-1].split('.')[0]) + 1

            inphaseinfo = pydicom.read_file(os.path.join(data_root, caseinphaseimg[i]))
            inphase = inphaseinfo.pixel_array
            voxelspacing[0] = float(inphaseinfo.PixelSpacing[0])
            voxelspacing[1] = float(inphaseinfo.PixelSpacing[1])
            voxelspacing[2] = float(inphaseinfo.SliceThickness)
            outphase = pydicom.read_file(os.path.join(data_root, caseoutphaseimg[i])).pixel_array
            inphase = Image.fromarray(inphase)
            if inphase.mode != 'RGB':
                inphase = inphase.convert('RGB')
            outphase = Image.fromarray(outphase)
            if outphase.mode != 'RGB':
                outphase = outphase.convert('RGB')
            inphase = np.array(inphase)
            inphase = torch.from_numpy(np.array(inphase).transpose(2,0,1)).float() / float(255)
            inmean = inphase.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            instd = inphase.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            inphase = inphase.sub(inmean).div(instd)
            outphase = np.array(outphase)
            outphase = torch.from_numpy(np.array(outphase).transpose(2,0,1)).float() / float(255)
            outmean = outphase.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            outstd = outphase.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            outphase = outphase.sub(outmean).div(outstd)
            mask = Image.open(os.path.join(data_root, casemask[i]))
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_arr = np.array(mask)
            mask_arr = np.expand_dims(mask_arr, axis=2)
            mask_arr = one_hot_mask(mask_arr, palette)
            mask = mask_arr[:,:,1]
            target.append(mask)

            with torch.no_grad():
                inphase = torch.unsqueeze(inphase.to(device), 0)
                outphase = torch.unsqueeze(outphase.to(device), 0)
                output = net(inphase, outphase)
                output = F.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)
                output = output.squeeze().cpu().numpy()
                generatedtarget.append(output)
                imgsavename.append(os.path.join(result_save_folder, casemask[i].split('/')[-1]))
        target = np.stack(target, axis=-1)
        generatedtarget = np.stack(generatedtarget, axis=-1)
        generatedtarget_keeplargest = keep_largeste_connected_components(generatedtarget)

        for i in np.arange(generatedtarget_keeplargest.shape[-1]):
            output_pil = generatedtarget_keeplargest[:,:,i] * 63
            output_pil = Image.fromarray(output_pil.astype(np.uint8), 'L')
            output_pil.save(imgsavename[i])
        dice = Dice3d_fn(generatedtarget_keeplargest, target)
        iou = IoU3d_fn(generatedtarget_keeplargest, target)
        TP, TN, FP, FN = TP_TN_FP_FN3d(generatedtarget_keeplargest, target)
        all_result.append([caseidx, dice, iou, TP, TN, FP, FN])
    result_csv = pd.DataFrame(all_result, columns=['Patient_case', 'Dice', 'IoU', 'TP', 'TN', 'FP', 'FN'])
    result_csv.to_csv(os.path.join(result_dir, result_csv_name), index=False)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_root = '../../inputs_chaos/All_Sets'
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    casecsv_file = '../../inputs_chaos/All_Sets_split/splitcases/val_data_10cases.csv'
    imgcsv_file = '../../inputs_chaos/All_Sets_split/splitimages_cleanlabel/val_data_10cases.csv'
    checkpoint_path = 'exampletrainedmodelsfortesting'
    checkpoint_binary1files = os.listdir(checkpoint_path)
    checkpoint_binary1files = [file for file in checkpoint_binary1files if file.endswith('.pkl')]

    result_root = 'examplesegmentationresults'
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    for checkpoint_file in checkpoint_binary1files:
        model_name = checkpoint_file.split('_')[0]
        checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_file))['net']
        result_dir = os.path.join(result_root, '{}_{}'.
                                  format(model_name, checkpoint_file.split('_')[-1].split('.')[0],
                                         ))
        result_csv_name = '{}.csv'.format(checkpoint_file.split('.pkl')[0])
        eval(model_name, num_classes, device, data_root, casecsv_file,
             imgcsv_file, checkpoint, result_dir, result_csv_name)