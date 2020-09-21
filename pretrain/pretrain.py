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
            default=90,
            help="number of epochs")
    
    parser.add_argument('-w', '--weight-path', type=str,
            default='./weight/',
            help="pre-trained weight path")

    parser.add_argument('--train-path', type=str,
            default='/disk1/datasets/img_type_datsets/ImageNet-1K/ILSVRC2012_img_train/',
            help='train dataset path')

    parser.add_argument('--val-path', type=str,
            default='/disk1/datasets/img_type_datsets/ImageNet-1K/ILSVRC2012_img_val/',
            help='val dataset path')

    parser.add_argument('--lr', type=float,
            default=0.01,
            help="Learning rate, deafult = 0.1")

    parser.add_argument('--gpu', type=int,
            default=-1)

    parser.add_argument('-d', '--weight-decay', type=float,
            default=1e-4, 
            help="Weight decay")

    parser.add_argument('-m', '--momentum', type=float,
            default=0.9,
            help='Momentum')

    parser.add_argument('--batch-size', type=int,
            default=256,
            help='Batch size')

    parser.add_argument('--log_dir', type=str,
            default='./log/',
            help='log dir')

    parser.add_argument('--resume', type=str,
            default=None,
            help='checkpoint path')

    parser.add_argument('--workers', type=int,
            default=8,
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
        device = torch.device('cuda')

        if args.gpu != -1:
            device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[34])

    # Load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print (f"loading checkpoint [{args.resume}]")

            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print (f"\nLoaded checkpoint at epoch {start_epoch}!\n")

        else:
            print ("Checkpoint file not found!\n")

        print ("--------------------------------------")

        cudnn.benchmark = True

    
    print ("\nLoading Dataset...")

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

    print ("\nLoaded datasets\n")
    print ("====================================\n")

    start_time = datetime.now().strftime('%m_%d__%H_%M')
    log_file_name = os.path.join('./log/', start_time + ".txt")

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01

    for epoch in range(start_epoch, args.epochs):

        train(train_loader, device, model, criterion, optimizer, epoch, args)

        acc1, loss = validate(val_loader, device, model, criterion, epoch, args)

        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)

        scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_file)

        if is_best:
            print (f"\nSave best model at acc: {acc1:.4f},  loss: {loss:.4f}!")

        logging(acc1, loss, epoch, is_best, log_file_name, get_lr(optimizer))

        print ("\n========================================\n")    


def train(train_loader, device, model, criterion, optimizer, epoch, args):
    model.train()
    running_loss = 0.0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc1, acc5 = accuracy(output, target, topk=(1,5))

        if is_print(i, len(train_loader)):
            print (f"Epoch [{epoch+1}/{args.epochs}] | Train iter [{i+1}/{len(train_loader)}] | acc1 = {acc1[0]:.3f} | acc5 = {acc5[0]:.3f} | loss = {(running_loss / float(i+1)):.5f} | lr = {get_lr(optimizer):.5f}")

def validate(val_loader, device, model, criterion, epoch, args):
    model.eval()
    running_loss = 0.0

    total_acc1 = 0.0
    total_acc5 = 0.0

    with torch.no_grad():

        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            acc1, acc5 = accuracy(output, target, topk=(1,5))

            total_acc1 += acc1
            total_acc5 += acc5

        total_acc1 /= len(val_loader)
        total_acc5 /= len(val_loader)
        print (f"\nEpoch [{epoch+1}/{args.epochs}] | Validation | acc1 = {total_acc1[0]:.3f} | acc5 = {total_acc5[0]:.3f} | loss = {(running_loss / float(i)):.5f}")

    return total_acc1[0], (running_loss / float(i))

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    path = log_dir + "_" + filename

    torch.save(state, path)

    if is_best:
        path_best = log_dir + "_" +  'model_best.pth.tar'
        shutil.copyfile(path, path_best)

def get_lr(optimizer):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break

    return lr

def logging(acc, loss, epoch, is_best, file_name, lr):
    with open(file_name, 'a+') as f:
        if is_best:
            f.write(f'Epoch {epoch+1} | Acc: {acc:.4f} | Val Loss: {loss:.4f} | LR: {lr:.4f} | best\n')
        else:
            f.write(f'Epoch {epoch+1} | Acc: {acc:.4f} | Val Loss: {loss:.4f} | LR: {lr:.4f}\n')

def accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _,pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res

def is_print(epoch, len_loader):
    if epoch <= 1000:
        if (epoch % 200 == 199):
            return True

    elif epoch > 1000:
        if (epoch % 2000 == 1999):
            return True

    elif epoch == (len_loader - 1):
        return True

    return False

if __name__ == '__main__':
    main()
