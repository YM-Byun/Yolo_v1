print ("Importing pytorch now...\n")

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import time
import shutil

from model import Yolo_pretrain
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int,
            default=135,
            help="number of epochs")
    
    parser.add_argument('-w', '--weight-path', type=str,
            help="pre-trained weight path")

    parser.add_argument('--train-path', type=str,
            default='./dataset/train',
            help='train dataset path')

    parser.add_argument('--val-path', type=str,
            default='./dataset/val',
            help='val dataset path')

    parser.add_argument('--lr', type=float,
            default=0.001,
            help="Learning rate, deafult = 0.001")

    parser.add_argument('-d', '--weight-decay', type=float,
            default=0.0005, 
            help="Weight decay")

    parser.add_argument('-m', '--momentum', type=float,
            default=0.9,
            help='Momentum')

    parser.add_argument('--batch-size', type=int,
            default=64,
            help='Batch size')

    parser.add_argument('--log_dir', type=str,
            default='./log/',
            help='log dir')

    parser.add_argument('--resume', type=str,
            default=None,
            help='checkpoint path')

    parser.add_argument('--workers', type=int,
            default=12,
            help="num of workers")


    return parser


def main():
    parser = get_argument_parser()

    args = parser.parse_args()

    log_file = args.log_dir + datetime.now().strftime('%m%d_%H:%M:%S')

    model = Yolo_pretrain()
    best_acc = 0
    start_epoch = 0
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        model.cuda()

    model.features = torch.nn.DataParallel(model.features)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    # Load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print (f"loading checkpoint [{args.resume}]")

            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dcit'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print (f"\nLoaded checkpoint at epoch {args.epochs}!\n")

        else:
            print ("Checkpoint file not found!\n")

        print ("--------------------------------------")

        cudnn.benchmark = True


    # Load datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(
            args.train_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]))

    val_dataset = ImageFolder(
            args.val_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize]))

    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print ("\nLoaded datasets")

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        train(train_loader, model, criterion, optimizer, epoch, args)

        acc1= validate(val_loader, model, criterion, epoch, args)

        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_file)


def adjust_learning_rate(optimizer, epoch, args):

    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Compute & store the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    isPrint = False

    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(None, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i < 1000:
            if i % 100 == 0:
                isPrint = True
        elif i == len(train_loader) - 1:
            isPrint = True

        else:
            if i % 500 == 0:
                isPrint = True

        if isPrint:
            print (f"Epoch [{epoch}] [{i+1} / {len(train_loader)}] ")
            print (f"\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})")
            print (f"\tData {data_time.val:.3f} ({data_time.avg:.3f})")
            print (f"\tLoss {losses.val:.4f} ({losses.avg:.4f})")
            print (f"\tAcc1 {top1.val:.3f} ({top1.avg:.3f})")
            print (f"\tAcc5 {top5.val:.3f} ({top5.avg:.3f})\n\n")

            isPrint = False

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    isPrint = False

    with torch.no_grad():
        end = time.time()

        for i, (data, target) in enumerate(val_loader):
            target = target.cuda(None, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i < 1000:
            if i % 100 == 0:
                isPrint = True
        elif i == len(val_loader) - 1:
            isPrint = True

        else:
            if i % 500 == 0:
                isPrint = True

        if isPrint:
            print (f"Test [{epoch}] [{i+1} / {len(val_loader)}]")
            print (f"\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})")
            print (f"\tLoss {losses.val:.4f} ({losses.avg:.4f})")
            print (f"\tAcc1 {top1.val:.3f} ({top1.avg:.3f})")
            print (f"\tAcc5 {top5.val:.3f} ({top5.avg:.3f})\n")
            isPrint = False

    print ("==========================================\n")

    return top1.avg

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    path = log_dir + "_" + filename

    torch.save(state, path)

    if is_best:
        path_best = log_dir + "_" +  'model_best.pth.tar'
        shutil.copyfile(path, path_best)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res

if __name__ == '__main__':
    main()
