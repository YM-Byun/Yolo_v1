import argparse
import math
import os
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import datetime

from yolo_v1 import Yolo_v1
from loss import Loss
from pretrain.model import Yolo_pretrain
from VOCDataset import VOCDataset

S = 0
B = 0
C = 0
init_lr = 0.001
base_lr = 0.001

import warnings
warnings.filterwarnings("ignore")

device = ""

def set_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int,
            default=135,
            help="number of epochs")
    
    parser.add_argument('-s', type=int,
            default=7,
            help="grid count, deafult = 7")

    parser.add_argument('-b', type=int,
            default=2,
            help="number of boundary box, deafult = 2")

    parser.add_argument('-w', '--pretrain-weight-path', type=str,
            default='./pretrain/weight/best.pth',
            help="pre trained weight path")

    parser.add_argument('-d', '--weight-decay', type=float,
            default=0.0005, 
            help="Weight decay")

    parser.add_argument('-m', '--momentum', type=float,
            default=0.9,
            help='Momentum')

    parser.add_argument('--batch-size', type=int,
            default=64,
            help='Batch size')

    parser.add_argument('--weight', type=str,
            default="./weight",
            help="weight path")

    parser.add_argument('--log', type=str,
            default="./log",
            help="log path")

    parser.add_argument('--resume', type=bool,
            default=False)

    parser.add_argument('--gpu', type=int,
            default=-1,
            help='Set specific GPU number')

    args = parser.parse_args()

    return args


def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)

    elif epoch == 1:
        lr = base_lr

    elif epoch == 75:
        lr = 0.001

    elif epoch == 105:
        lr = 0.0001

    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def isPrint(i, length):
    isPrint = False

    if i == length - 1:
        isPrint = True

    else:
        if i % 60 == 59:
            isPrint = True

    return isPrint

def save_checkpoint(state, epoch, result_dir):
    filenames = os.listdir(result_dir)

    for name in filenames:
        if name.startswith('checkpoint'):
            os.remove(os.path.join(result_dir, name))

    checkpoint_file = os.path.join(result_dir , "checkpoint_" + str(epoch + 1) + ".pth")
    torch.save(state, checkpoint_file)

def train(train_loader, yolo, criterion, optimizer, epoch, args):
    global device

    yolo.train()
    running_loss = 0.0

    for i, (imgs, targets) in enumerate(train_loader):
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        
        if is_cuda:
            imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if isPrint(i, len(train_loader)):
            print (f"Epoch [{epoch+1}/{args.epochs}] | Train coniter [{i+1}/{len(train_loader)}] | loss = {(running_loss / float(i+1)):.5f} | lr = {get_lr(optimizer):.5f}")

    return running_loss / float(i+1)

def val(val_loader, yolo, criterion, epoch, args):
    # val
    yolo.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            if is_cuda:
                imgs, targets = imgs.to(device), targets.to(device)

            preds = yolo(imgs)

            loss = criterion(preds, targets)
            running_loss += loss.item()

    print (f"Epoch [{epoch+1}/{args.epochs}] | Validation | loss = {(running_loss / len(val_loader)):.5f}")

    return running_loss / len(val_loader)


def logging(loss, epoch, is_best, file_name, lr):
    with open(file_name, 'a+') as f:
        if is_best:
            f.write(f'Epoch {epoch+1} | Val Loss: {loss:.4f} | LR: {lr:.4f} | best\n')
        else:
            f.write(f'Epoch {epoch+1} | Val Loss: {loss:.4f} | LR: {lr:.4f}\n')

if __name__ == "__main__":
    args = set_argument()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        if args.gpu != -1:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')


    train_dataset = VOCDataset(is_train=True, image_dir="./dataset/Image",
                label_txt='./dataset/voc_train.txt')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)


    val_dataset = VOCDataset(is_train=False, image_dir="./dataset/Image",
                label_txt='./dataset/voc_val.txt')

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)

    print ("\nLoaded dataset!\n")

    print ("=======================================")

    pretrain = Yolo_pretrain(conv_only=True, init_weight=True)

    if is_cuda:
        pretrain.features = pretrain.features.to(device)

    src_state_dict = torch.load(args.pretrain_weight_path, map_location='cuda:0')['state_dict']
    dst_state_dict = pretrain.state_dict()

    for key in dst_state_dict.keys():
        dst_state_dict[key] = src_state_dict[key]

    pretrain.load_state_dict(dst_state_dict)

    print ("\nLoaded pretrain weight\n")

    print ("======================================\n")


    yolo = Yolo_v1(pretrain.features)

    if is_cuda:
        yolo = yolo.to(device)

    # Loss func
    criterion = Loss()
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_time = datetime.datetime.now().strftime('%m_%d__%H_%M')

    log_file_name = os.path.join(args.log, start_time + ".txt")

    best_loss = np.inf

    start_epoch = 0

    # Load checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] - 1
            best_loss = checkpoint['best_loss']
            yolo.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print ("Loaded checkpoint!")
            print (f"Start epoch from {start_epoch + 1}")
            print (f"Lastest best val loss: {best_loss}\n")

    for epoch in range(start_epoch, args.epochs):
        train(train_loader, yolo, criterion, optimizer, epoch, args)

        print (" ")

        loss = val(val_loader, yolo, criterion, epoch, args)

        is_best = False

        if best_loss > loss:
            is_best = True
            best_loss = loss
        
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': yolo.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()},
            epoch+1, args.weight)

        if is_best:
            torch.save(yolo.state_dict(), args.weight + "/best_model.pth")
            print (f"\nSave best model at loss: {loss:.4f}!")

        logging(loss, epoch, is_best, log_file_name, get_lr(optimizer))

        print ("\n======================================\n")
        
