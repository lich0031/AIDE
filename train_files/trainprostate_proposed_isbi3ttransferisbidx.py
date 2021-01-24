import sys

sys.path.extend(['../'])
import os, time, argparse, random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
import pydicom
from skimage import measure
import torch.nn.functional as F
import shutil
import SimpleITK as sitk
from datasetprostate_proposed import prostate_seg, Compose, Resize, RandomRotate, RandomHorizontallyFlip, ToTensor, Normalize
from models_singlemodalinput import UNet, UNetsa
from utils import CrossEntropyLoss2d, DiceLoss, MulticlassDiceLoss, CEMDiceLossImage, PolyLR, \
    MulticlassDice_fn, MulticlassAccuracy_fn, Dice_fn, MulticlassMSELoss

def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for Prostate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='UNet', type=str, help='UNet, ...')
    parser.add_argument('--data_mean', default=None, nargs='+', type=float,
                        help='Normalize mean')
    parser.add_argument('--data_std', default=None, nargs='+', type=float,
                        help='Normalize std')
    parser.add_argument('--rotation', default=60, type=float, help='rotation angle')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='pretrain num epoch')
    parser.add_argument('--num_epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--loss', default='cedice', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=256, type=int, help='512')
    parser.add_argument('--temperature', default=1.0, type=float, help='0.5')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--cedice_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='weight for ce and dice loss')
    parser.add_argument('--segcor_weight', default=[1.0, 10.0], nargs='+', type=float,
                        help='weight for seg and pseudolabel seg')
    parser.add_argument('--ceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for ce loss')
    parser.add_argument('--diceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for dice loss')
    parser.add_argument('--checkpoint', default='checkpoint_train3tgeneratedx_comparisoncrossdomain/')
    parser.add_argument('--history', default='history_train3tgeneratedx_comparisoncrossdomain')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--repetition', default=100, type=int, help='...')

    args = parser.parse_args()
    return args

def record_params(args):
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Segmeantation for Prostate MR(Data: {}) \n'.format(localtime))
    logging.info('**************Parameters***************')

    args_dict = args.__dict__
    for key, value in args_dict.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('**************Parameters***************\n')

def build_model(model_name, num_classes):
    if model_name.lower() == 'unet':
        net = UNet(num_classes=num_classes)
    elif model_name.lower() == 'unetsa':
        net = UNetsa(num_classes=num_classes)
    else:
        raise ValueError('Model not implemented')
    return net

def reverseaug(augset, augoutput, classno):
    for batch_idx in range(len(augset['augno'])):
        for aug_idx in range(augset['augno'][batch_idx]):
            imgflip = augset['hflip{}'.format(aug_idx + 1)][batch_idx]
            rotation = 0 - augset['degree{}'.format(aug_idx + 1)][batch_idx]
            for classidx in range(classno):
                mask = augoutput[aug_idx][batch_idx, classidx, :, :]
                mask = mask.cpu().numpy()
                mask = Image.fromarray(mask, mode='F')
                if imgflip:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.rotate(rotation, Image.BILINEAR)
                mask = torch.from_numpy(np.array(mask))
                augoutput[aug_idx][batch_idx, classidx, :, :] = mask
    return augoutput

def sharpen(mask, temperature):
    masktemp = torch.pow(mask, temperature)
    masktempsum = masktemp.sum(dim=1).unsqueeze(dim=1)
    sharpenmask = masktemp / masktempsum
    return sharpenmask

def keep_largest_connected_components(mask):
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    blobs = measure.label(mask, connectivity=1)  # connectivity 1: 4 neighbours 2: 8 neighbours
    props = measure.regionprops(blobs)
    area = [ele.area for ele in props]
    if mask.max() > 0:
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = 1
    return out_img

def Dice3d_fn(inputs, targets):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = iflat * tflat
    intersection = 2 * np.sum(intersection)
    union = np.sum(iflat) + np.sum(tflat)
    if union == 0:
        if intersection == 0:
            dice_image = 1
        else:
            dice_image = 0
    else:
        dice_image = intersection / union
    return dice_image

def makefolder(folderpath):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

def copy_files(source_dir, target_ir):
    for folder in os.listdir(source_dir):
        source_folder = os.path.join(source_dir, folder)
        save_folder = os.path.join(target_ir, folder)
        makefolder(save_folder)
        for file in os.listdir(source_folder):
            source_file = os.path.join(source_folder, file)
            if os.path.isfile(source_file):
                shutil.copy(source_file, save_folder)

def Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv, labelcase_csv, tempmaskfolder):
    makefolder(os.path.join(train_root, tempmaskfolder))

    # parameters
    args = parse_args()

    # record
    record_params(args)

    train_cases = pd.read_csv(traincase_csv)['Image'].tolist()
    train_masks = pd.read_csv(traincase_csv)['Mask'].tolist()
    test_cases = pd.read_csv(testcase_csv)['Image'].tolist()
    label_cases = pd.read_csv(labelcase_csv)['Image'].tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_order
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.torch_seed)
    random.seed(args.torch_seed)

    if args.cudnn == 0:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2

    net1 = build_model(args.model_name, num_classes)
    net2 = build_model(args.model_name, num_classes)
    params1_name = '{}_temp{}_r{}_net1.pkl'.format(args.model_name, args.temperature, args.repetition)
    params2_name = '{}_temp{}_r{}_net2.pkl'.format(args.model_name, args.temperature, args.repetition)
    start_epoch = 0
    end_epoch = args.num_epoch

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)
    net1.to(device)
    net2.to(device)

    # data
    train_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        RandomRotate(args.rotation),
        RandomHorizontallyFlip(),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])
    test_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])

    train_dataset = prostate_seg(root=train_root, csv_file=train_csv, tempmaskfolder=tempmaskfolder, transform=train_aug)
    test_dataset = prostate_seg(root=train_root, csv_file=test_csv, tempmaskfolder=tempmaskfolder, transform=test_aug)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, shuffle=False)

    # loss function, optimizer and scheduler
    cedice_weight = torch.tensor(args.cedice_weight)
    ceclass_weight = torch.tensor(args.ceclass_weight)
    diceclass_weight = torch.tensor(args.diceclass_weight)

    if args.loss == 'ce':
        criterion = CrossEntropyLoss2d(weight=ceclass_weight).to(device)
    elif args.loss == 'dice':
        criterion = MulticlassDiceLoss(weight=diceclass_weight).to(device)
    elif args.loss == 'cedice':
        criterion = CEMDiceLossImage(cediceweight=cedice_weight, ceclassweight=ceclass_weight,
                                     diceclassweight=diceclass_weight).to(device)
    else:
        print('Do not have this loss')
    corrlosscriterion = MulticlassMSELoss(reduction='none').to(device)

    # define augmentation loss effect schedule
    rate_schedule = np.ones(args.num_epoch)

    optimizer1 = Adam(net1.parameters(), lr=args.lr, amsgrad=True)
    optimizer2 = Adam(net2.parameters(), lr=args.lr, amsgrad=True)

    ## scheduler
    if args.lr_policy == 'StepLR':
        scheduler1 = StepLR(optimizer1, step_size=30, gamma=0.5)
        scheduler2 = StepLR(optimizer2, step_size=30, gamma=0.5)
    if args.lr_policy == 'PolyLR':
        scheduler1 = PolyLR(optimizer1, max_epoch=end_epoch, power=0.9)
        scheduler2 = PolyLR(optimizer2, max_epoch=end_epoch, power=0.9)

    # training process
    logging.info('Start Training For Prostate Seg')
    besttraincasedice = 0.0
    for epoch in range(start_epoch, end_epoch):

        ts = time.time()
        rate_schedule[epoch] = min((float(epoch) / float(args.warmup_epoch)) ** 2, 1.0)

        # train
        net1.train()
        net2.train()

        train_loss1 = 0.
        train_dice1 = 0.
        train_count = 0
        train_loss2 = 0.
        train_dice2 = 0.

        for batch_idx, (inputs, augset, targets, targets1, targets2) in \
                tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):

            augoutput1 = []
            augoutput2 = []
            for aug_idx in range(augset['augno'][0]):
                augimg = augset['img{}'.format(aug_idx + 1)].to(device)
                augoutput1.append(net1(augimg).detach())
                augoutput2.append(net2(augimg).detach())

            augoutput1 = reverseaug(augset, augoutput1, classno=num_classes)
            augoutput2 = reverseaug(augset, augoutput2, classno=num_classes)

            for aug_idx in range(augset['augno'][0]):
                augmask1 = torch.nn.functional.softmax(augoutput1[aug_idx], dim=1)
                augmask2 = torch.nn.functional.softmax(augoutput2[aug_idx], dim=1)

                if aug_idx == 0:
                    pseudo_label1 = augmask1
                    pseudo_label2 = augmask2
                else:
                    pseudo_label1 += augmask1
                    pseudo_label2 += augmask2

            pseudo_label1 = pseudo_label1 / float(augset['augno'][0])
            pseudo_label2 = pseudo_label2 / float(augset['augno'][0])
            pseudo_label1 = sharpen(pseudo_label1, args.temperature)
            pseudo_label2 = sharpen(pseudo_label2, args.temperature)
            weightmap1 = 1.0 - 4.0 * pseudo_label1[:, 0, :, :] * pseudo_label1[:, 1, :, :]
            weightmap1 = weightmap1.unsqueeze(dim=1)
            weightmap2 = 1.0 - 4.0 * pseudo_label2[:, 0, :, :] * pseudo_label2[:, 1, :, :]
            weightmap2 = weightmap2.unsqueeze(dim=1)

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            loss1_segpre = criterion(outputs1, targets2)
            loss2_segpre = criterion(outputs2, targets1)
            _, indx1 = loss1_segpre.sort()
            _, indx2 = loss2_segpre.sort()
            loss1_seg1 = criterion(outputs1[indx2[0:2], :, :, :], targets2[indx2[0:2], :, :]).mean()
            loss2_seg1 = criterion(outputs2[indx1[0:2], :, :, :], targets1[indx1[0:2], :, :]).mean()
            loss1_seg2 = criterion(outputs1[indx2[2:], :, :, :], targets2[indx2[2:], :, :]).mean()
            loss2_seg2 = criterion(outputs2[indx1[2:], :, :, :], targets1[indx1[2:], :, :]).mean()
            loss1_cor = weightmap2[indx2[2:], :, :, :] * corrlosscriterion(outputs1[indx2[2:], :, :, :],
                                                                           pseudo_label2[indx2[2:], :, :, :])
            loss1_cor = loss1_cor.mean()
            loss1 = args.segcor_weight[0] * (loss1_seg1 + (1.0 - rate_schedule[epoch]) * loss1_seg2) + \
                    args.segcor_weight[1] * rate_schedule[epoch] * loss1_cor

            loss2_cor = weightmap1[indx1[2:], :, :, :] * corrlosscriterion(outputs2[indx1[2:], :, :, :],
                                                                           pseudo_label1[indx1[2:], :, :, :])
            loss2_cor = loss2_cor.mean()
            loss2 = args.segcor_weight[0] * (loss2_seg1 + (1.0 - rate_schedule[epoch]) * loss2_seg2) + \
                    args.segcor_weight[1] * rate_schedule[epoch] * loss2_cor
            loss1.backward(retain_graph=True)
            optimizer1.step()
            loss2.backward()
            optimizer2.step()
            train_count += inputs.shape[0]
            train_loss1 += loss1.item() * inputs.shape[0]
            train_dice1 += Dice_fn(outputs1, targets2).item()
            train_loss2 += loss2.item() * inputs.shape[0]
            train_dice2 += Dice_fn(outputs2, targets1).item()
        train_loss1_epoch = train_loss1 / float(train_count)
        train_dice1_epoch = train_dice1 / float(train_count)
        train_loss2_epoch = train_loss2 / float(train_count)
        train_dice2_epoch = train_dice2 / float(train_count)

        print(rate_schedule[epoch])
        print(args.segcor_weight[0] * (loss1_seg1 + (1.0 - rate_schedule[epoch]) * loss1_seg2))
        print(args.segcor_weight[1] * rate_schedule[epoch] * loss1_cor)

        print(args.segcor_weight[0] * (loss2_seg1 + (1.0 - rate_schedule[epoch]) * loss2_seg2))
        print(args.segcor_weight[1] * rate_schedule[epoch] * loss2_cor)

        # test
        net1.eval()
        net2.eval()
        test_loss1 = 0.
        test_dice1 = 0.
        test_loss2 = 0.
        test_dice2 = 0.
        test_count = 0
        for batch_idx, (inputs, augset, targets, targets1, targets2) in \
                tqdm(enumerate(test_loader), total=int(len(test_loader.dataset) / args.batch_size)):
            with torch.no_grad():
                inputs = inputs.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                loss1 = criterion(outputs1, targets2).mean()
                loss2 = criterion(outputs2, targets1).mean()
            test_count += inputs.shape[0]
            test_loss1 += loss1.item() * inputs.shape[0]
            test_dice1 += Dice_fn(outputs1, targets2).item()
            test_loss2 += loss2.item() * inputs.shape[0]
            test_dice2 += Dice_fn(outputs2, targets1).item()

        test_loss1_epoch = test_loss1 / float(test_count)
        test_dice1_epoch = test_dice1 / float(test_count)
        test_loss2_epoch = test_loss2 / float(test_count)
        test_dice2_epoch = test_dice2 / float(test_count)

        testcasedices1 = torch.zeros(len(test_cases))
        testcasedices2 = torch.zeros(len(test_cases))
        startimgslices = torch.zeros(len(test_cases))
        for casecount in tqdm(range(len(test_cases)), total=len(test_cases)):
            caseidx = test_cases[casecount]
            caseimg = [file for file in test_dataset.imgs if caseidx.split('/')[-1].split('.')[0] in file]
            caseimg.sort()
            casemask = [file for file in test_dataset.masks if caseidx.split('/')[-1].split('.')[0] in file]
            casemask.sort()
            generatedtarget1 = []
            generatedtarget2 = []
            target1 = []
            target2 = []
            startcaseimg = int(torch.sum(startimgslices[:casecount + 1]))
            for imgidx in range(len(caseimg)):
                assert caseimg[imgidx].split('/')[-1].split('.')[0] == \
                       casemask[imgidx].split('/')[-1].split('.')[0].split('_')[0]
                sample = test_dataset.__getitem__(imgidx + startcaseimg)
                input = sample[0]
                mask1 = sample[3]
                mask2 = sample[4]
                target1.append(mask1)
                target2.append(mask2)
                with torch.no_grad():
                    input = torch.unsqueeze(input.to(device), 0)
                    output1 = net1(input)
                    output1 = F.softmax(output1, dim=1)
                    output1 = torch.argmax(output1, dim=1)
                    output1 = output1.squeeze().cpu().numpy()
                    generatedtarget1.append(output1)
                    output2 = net2(input)
                    output2 = F.softmax(output2, dim=1)
                    output2 = torch.argmax(output2, dim=1)
                    output2 = output2.squeeze().cpu().numpy()
                    generatedtarget2.append(output2)
            target1 = np.stack(target1, axis=-1)
            target2 = np.stack(target2, axis=-1)
            generatedtarget1 = np.stack(generatedtarget1, axis=-1)
            generatedtarget2 = np.stack(generatedtarget2, axis=-1)
            generatedtarget1_keeplargest = keep_largest_connected_components(generatedtarget1)
            generatedtarget2_keeplargest = keep_largest_connected_components(generatedtarget2)
            testcasedices1[casecount] = Dice3d_fn(generatedtarget1_keeplargest, target1)
            testcasedices2[casecount] = Dice3d_fn(generatedtarget2_keeplargest, target2)
            if casecount + 1 < len(test_cases):
                startimgslices[casecount + 1] = len(caseimg)
        testcasedice1 = testcasedices1.sum() / float(len(test_cases))
        testcasedice2 = testcasedices2.sum() / float(len(test_cases))

        traincasedices1 = torch.zeros(len(train_cases))
        traincasedices2 = torch.zeros(len(train_cases))
        # update pseudolabel
        startimgslices = torch.zeros(len(train_cases))
        generatedmask1 = []
        generatedmask2 = []
        for casecount in tqdm(range(len(train_cases)), total=len(train_cases)):
            caseidx = train_cases[casecount]
            caseimg = [file for file in train_dataset.imgs if caseidx.split('/')[-1].split('.')[0] in file]
            caseimg.sort()
            casemask = [file for file in train_dataset.masks if caseidx.split('/')[-1].split('.')[0] in file]
            casemask.sort()
            generatedtarget1 = []
            generatedtarget2 = []
            target1 = []
            target2 = []
            startcaseimg = int(torch.sum(startimgslices[:casecount + 1]))
            for imgidx in range(len(caseimg)):
                assert caseimg[imgidx].split('/')[-1].split('.')[0] == \
                       casemask[imgidx].split('/')[-1].split('.')[0].split('_')[0]
                sample = train_dataset.__getitem__(imgidx + startcaseimg)
                input = sample[0]
                mask1 = sample[3]
                mask2 = sample[4]
                target1.append(mask1)
                target2.append(mask2)
                with torch.no_grad():
                    input = torch.unsqueeze(input.to(device), 0)
                    output1 = net1(input)
                    output1 = F.softmax(output1, dim=1)
                    output1 = torch.argmax(output1, dim=1)
                    output1 = output1.squeeze().cpu().numpy()
                    generatedtarget1.append(output1)

                    output2 = net2(input)
                    output2 = F.softmax(output2, dim=1)
                    output2 = torch.argmax(output2, dim=1)
                    output2 = output2.squeeze().cpu().numpy()
                    generatedtarget2.append(output2)

            target1 = np.stack(target1, axis=-1)
            target2 = np.stack(target2, axis=-1)
            generatedtarget1 = np.stack(generatedtarget1, axis=-1)
            generatedtarget2 = np.stack(generatedtarget2, axis=-1)
            generatedtarget1_keeplargest = keep_largest_connected_components(generatedtarget1)
            generatedtarget2_keeplargest = keep_largest_connected_components(generatedtarget2)
            traincasedices1[casecount] = Dice3d_fn(generatedtarget1_keeplargest, target1)
            traincasedices2[casecount] = Dice3d_fn(generatedtarget2_keeplargest, target2)
            generatedmask1.append(generatedtarget1_keeplargest)
            generatedmask2.append(generatedtarget2_keeplargest)
            if casecount + 1 < len(train_cases):
                startimgslices[casecount + 1] = len(caseimg)

        traincasedice1 = traincasedices1.sum() / float(len(train_cases))
        traincasedice2 = traincasedices2.sum() / float(len(train_cases))

        traincasediceavgtemp = (traincasedice1 + traincasedice2) / 2.0

        if traincasediceavgtemp > besttraincasedice:
            backfolder = os.path.join(train_root, tempmaskfolder + '_besttraindice')
            if os.path.exists(backfolder):
                shutil.rmtree(backfolder)
            shutil.copytree(os.path.join(train_root, tempmaskfolder), backfolder)

            besttraincasedice = traincasediceavgtemp
            logging.info('Best Checkpoint {} Saving...'.format(epoch + 1))

            save_model = net1
            if torch.cuda.device_count() > 1:
                save_model = list(net1.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss1_epoch,
                'epoch': epoch + 1,
            }
            savecheckname = os.path.join(args.checkpoint, params1_name.split('.pkl')[0] +
                                         '_besttraincasedice.' + params1_name.split('.')[-1])
            torch.save(state, savecheckname)

            save_model = net2
            if torch.cuda.device_count() > 1:
                save_model = list(net2.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss2_epoch,
                'epoch': epoch + 1,
            }
            savecheckname = os.path.join(args.checkpoint, params2_name.split('.pkl')[0] +
                                         '_besttraincasedice.' + params2_name.split('.')[-1])
            torch.save(state, savecheckname)

        if (epoch + 1) <= args.warmup_epoch or (epoch + 1) % 10 == 0:
            traincasediceavg = traincasediceavgtemp
            selected_samples = int(0.25 * (len(train_cases) - len(label_cases)))
            save_root = os.path.join(train_root, tempmaskfolder)
            _, sortidx1 = traincasedices1.sort()
            selectedidxs = sortidx1[:selected_samples]
            for selectedidx in selectedidxs:
                caseidx = train_cases[selectedidx]
                if caseidx not in label_cases:
                    save_name = os.path.join(save_root, '{}_net1.{}'.format(
                        train_masks[selectedidx].split('/')[-1].split('.')[0],
                        train_masks[selectedidx].split('/')[-1].split('.')[-1]))
                    smasksave = sitk.GetImageFromArray(np.transpose(generatedmask1[selectedidx], [2,0,1]))
                    sitk.WriteImage(smasksave, save_name)
            logging.info('Mask {} modify for net1'.format([train_cases[i].split('/')[-1] for i in selectedidxs]))

            _, sortidx2 = traincasedices2.sort()
            selectedidxs = sortidx2[:selected_samples]
            for selectedidx in selectedidxs:
                caseidx = train_cases[selectedidx]
                if caseidx not in label_cases:
                    save_name = os.path.join(save_root, '{}_net2.{}'.format(
                        train_masks[selectedidx].split('/')[-1].split('.')[0],
                        train_masks[selectedidx].split('/')[-1].split('.')[-1]))
                    smasksave = sitk.GetImageFromArray(np.transpose(generatedmask2[selectedidx], [2,0,1]))
                    sitk.WriteImage(smasksave, save_name)
            logging.info('Mask {} modify for net2'.format([train_cases[i].split('/')[-1] for i in selectedidxs]))

        time_cost = time.time() - ts
        logging.info(
            'epoch[%d/%d]: train_loss1: %.3f | test_loss1: %.3f | train_dice1: %.3f | test_dice1: %.3f || '
            'traincase_dice1: %.3f || testcase_dice1: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, train_loss1_epoch, test_loss1_epoch, train_dice1_epoch, test_dice1_epoch,
               traincasedice1, testcasedice1, time_cost))
        logging.info(
            'epoch[%d/%d]: train_loss2: %.3f | test_loss2: %.3f | train_dice2: %.3f | test_dice2: %.3f || '
            'traincase_dice2: %.3f || testcase_dice2: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, train_loss2_epoch, test_loss2_epoch, train_dice2_epoch, test_dice2_epoch,
               traincasedice2, testcasedice2, time_cost))
        if args.lr_policy != 'None':
            scheduler1.step()
            scheduler2.step()

args = parse_args()
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
if not os.path.exists(args.history):
    os.mkdir(args.history)

log_name = '{}_temp{}_r{}.log'.format(args.model_name, args.temperature, args.repetition)
logging_save = os.path.join(args.history, log_name)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])

if __name__ == "__main__":
    args = parse_args()
    train_root = '../../inputs_prostatemr'
    train_csv = '../../inputs_prostatemr/Prostate_split2D_crossdomain/ISBI2013_nrrd_combineall/train3tgeneratedx_train.csv'
    test_csv = '../../inputs_prostatemr/Prostate_split2D_crossdomain/ISBI2013_nrrd_combineall/train3tgeneratedx_testall.csv'
    traincase_csv = '../../inputs_prostatemr/Prostate_split2D_crossdomain/ISBI2013_nrrd_combineall/train3tgeneratedx_casetrain.csv'
    testcase_csv = '../../inputs_prostatemr/Prostate_split2D_crossdomain/ISBI2013_nrrd_combineall/train3tgeneratedx_casetestall.csv'
    labeledcase_csv = '../../inputs_prostatemr/Prostate_split2D_crossdomain/ISBI2013_nrrd_combineall/train3tgeneratedx_labeledcasetrain.csv'
    tempmaskfolder = 'generated_masks_train3tgeneratedx'
    makefolder(os.path.join(train_root, tempmaskfolder))
    tempmaskfolder = 'generated_masks_train3tgeneratedx/{}_{}'.format(args.model_name, args.repetition)
    Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv, labeledcase_csv, tempmaskfolder)