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
from datasetbreast_proposed import breast_seg, Compose, Resize, RandomRotate, RandomHorizontallyFlip, ToTensor, Normalize
from models_singlemodalinput import UNet
from utils import CrossEntropyLoss2d, DiceLoss, MulticlassDiceLoss, CEMDiceLoss, PolyLR, \
    MulticlassDice_fn, MulticlassAccuracy_fn, Dice_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for Breast',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='UNet', type=str, help='UNet, ...')
    parser.add_argument('--data_mean', default=None, nargs='+', type=float,
                        help='Normalize mean')
    parser.add_argument('--data_std', default=None, nargs='+', type=float,
                        help='Normalize std')
    parser.add_argument('--rotation', default=60, type=float, help='rotation angle')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='1', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--loss', default='cedice', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=384, type=int, help='512')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--cedice_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='weight for ce and dice loss')
    parser.add_argument('--ceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for ce loss')
    parser.add_argument('--diceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for dice loss')
    parser.add_argument('--checkpoint', default='checkpoint_breastdata3_comparison272cases25labels/')
    parser.add_argument('--history', default='history_breastdata3_comparison272cases25labels')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--repetition', default=1, type=int, help='...')

    args = parser.parse_args()
    return args


def record_params(args):
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Segmeantation for Breast MR(Data: {}) \n'.format(localtime))
    logging.info('**************Parameters***************')

    args_dict = args.__dict__
    for key, value in args_dict.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('**************Parameters***************\n')


def build_model(model_name, num_classes):
    if model_name.lower() == 'unet':
        net = UNet(num_classes=num_classes)
    else:
        raise ValueError('Model not implemented')
    return net


def makefolder(folderpath):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)


def Dice2d(input, target):
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = iflat * tflat
    intersection = 2 * np.sum(intersection)
    union = np.sum(iflat) + np.sum(tflat)
    dice_image = intersection / union
    return dice_image


def Train(train_root, train_csv, test_csv):
    # parameters
    args = parse_args()

    # record
    record_params(args)

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

    net = build_model(args.model_name, num_classes)

    # resume
    params_name = '{}_r{}.pkl'.format(args.model_name, args.repetition)
    start_epoch = 0
    history = {'train_loss': [], 'test_loss': [],
               'train_dice': [], 'test_dice': []}
    end_epoch = start_epoch + args.num_epoch

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # data
    train_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])
    test_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])

    train_dataset = breast_seg(root=train_root, csv_file=train_csv, transform=train_aug)
    test_dataset = breast_seg(root=train_root, csv_file=test_csv, transform=test_aug)

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
        criterion = CEMDiceLoss(cediceweight=cedice_weight, ceclassweight=ceclass_weight,
                                diceclassweight=diceclass_weight).to(device)
    else:
        print('Do not have this loss')

    optimizer = Adam(net.parameters(), lr=args.lr, amsgrad=True)

    ## scheduler
    if args.lr_policy == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    if args.lr_policy == 'PolyLR':
        scheduler = PolyLR(optimizer, max_epoch=end_epoch, power=0.9)

    # training process
    logging.info('Start Training For Breast Seg')

    besttraindice = 0.

    for epoch in range(start_epoch, end_epoch):

        ts = time.time()
        net.train()

        for batch_idx, (imgs, _, targets, _, _) in tqdm(enumerate(train_loader),
                                                        total=int(len(train_loader.dataset) / args.batch_size)):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # test
        net.eval()
        test_loss = 0.
        test_dice = 0.
        test_count = 0
        for batch_idx, (imgs, _, targets, _, _) in tqdm(enumerate(test_loader),
                                                        total=int(len(test_loader.dataset) / args.batch_size)):
            with torch.no_grad():
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs)
                loss = criterion(outputs, targets).mean()
            test_count += imgs.shape[0]
            test_loss += loss.item() * imgs.shape[0]
            test_dice += Dice_fn(outputs, targets).item()

        test_loss_epoch = test_loss / float(test_count)
        test_dice_epoch = test_dice / float(test_count)
        history['test_loss'].append(test_loss_epoch)
        history['test_dice'].append(test_dice_epoch)
        train_loss = 0.
        train_dice = 0.
        train_count = 0
        for batch_idx, (imgs, _, targets, _, _) in tqdm(enumerate(train_loader),
                                                        total=int(len(train_loader.dataset) / args.batch_size)):
            with torch.no_grad():
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs)
                loss = criterion(outputs, targets).mean()
            train_count += imgs.shape[0]
            train_loss += loss.item() * imgs.shape[0]
            train_dice += Dice_fn(outputs, targets).item()
        train_loss_epoch = train_loss / float(train_count)
        train_dice_epoch = train_dice / float(train_count)
        history['train_loss'].append(train_loss_epoch)
        history['train_dice'].append(train_dice_epoch)

        time_cost = time.time() - ts
        logging.info(
            'epoch[%d/%d]: train_loss: %.3f | test_loss: %.3f | train_dice: %.3f | test_dice: %.3f || time: %.1f'
            % (epoch + 1, end_epoch, train_loss_epoch, test_loss_epoch, train_dice_epoch, test_dice_epoch, time_cost))

        if args.lr_policy != 'None':
            scheduler.step()

        # save checkpoint
        if train_dice_epoch > besttraindice:
            besttraindice = train_dice_epoch
            logging.info('Besttraindice Checkpoint {} Saving...'.format(epoch + 1))

            save_model = net
            if torch.cuda.device_count() > 1:
                save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss_epoch,
                'dice': test_dice_epoch,
                'epoch': epoch + 1,
                'history': history
            }
            savecheckname = os.path.join(args.checkpoint, params_name.split('.pkl')[0] +
                                         '_besttraindice.' + params_name.split('.')[-1])
            torch.save(state, savecheckname)

args = parse_args()
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
if not os.path.exists(args.history):
    os.mkdir(args.history)

log_name = '{}_r{}.log'.format(args.model_name, args.repetition)
logging_save = os.path.join(args.history, log_name)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])

if __name__ == "__main__":
    args = parse_args()
    train_root = '../../inputs_breastMR_Henan_372cases'
    train_csv = '../../inputs_breastMR_Henan_372cases/BreastMR_csvfiles/splitnoisylabels/train_data_25cases_imgs.csv'
    test_csv = '../../inputs_breastMR_Henan_372cases/BreastMR_csvfiles/splitcleanlabels/val_data_100cases_imgs.csv'
    Train(train_root, train_csv, test_csv)