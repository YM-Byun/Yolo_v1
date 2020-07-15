import argparse
import math
import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from yolo_v1 import Yolo_v1
from loss import Yolo_loss
from pre_train.model import Yolo_pretrain

S = 0
B = 0
C = 0
decay = 0.0005
momentum = 0.9
batch_size = 64
num_epochs = 135
init_lr = 0.001
base_lr = 0.01

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
            default='pre_train/weight/model_best.pth.tar',
            help="pre-trained weight path")

    parser.add_argument('--train-path', type=str,
            default='./dataset/train',
            help='train dataset path')

    parser.add_argument('--val-path', type=str,
            default='./dataset/val',
            help='validation dataset path')

    parser.add_argument('-d', '--weight-decay', type=float,
            default=0.0005, 
            help="Weight decay")

    parser.add_argument('-m', '--momentum', type=float,
            default=0.9,
            help='Momentum')

    parser.add_argument('--batch-size', type=int,
            default=64,
            help='Batch size')

    parser.add_argument('--result-path', type=str,
            default="./result",
            help="result path")

    args = parser.parse_args()

    return args


def update_lr(optimizer, epoch, burin_base, burin_exp=4.0):
    if epoch == 0:
        lr = learning_rate + (base_lr - init_lr) * math.pow(burin_base, burnin_exp)

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


def isPrint(i):
    isPrint = False

    if i < 1000:
        if i % 100 == 0:
            isPrint = True

    elif i == len(train_2007_loader) - 1:
        isPrint = True

    else:
        if i % 500 == 0:
            isPrint = True

    return isPrint


if __name__ == "__main__":
    args = set_argument()

    is_cuda = torch.cuda.is_available()

    normalize = transforms.Normalize(mean=[0.485, 0.485, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.RandomHorizontalFlip()])

    train_2012_dataset = datasets.VOCSegmentation(root='dataset/train', year='2012', image_set='train', download=True, transforms=transform)

    train_2012_loader = DataLoader(train_2012_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    print ("loaded 2012 train dataset\n")

    val_2012_dataset = datasets.VOCSegmentation(root='dataset/val', year='2012', image_set='val', download=True, transforms=transform)

    val_2012_loader = DataLoader(val_2012_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    print ("loaded 2012 val dataset\n")

    train_2007_dataset = datasets.VOCSegmentation(root='dataset/train', year='2007', image_set='train', download=True, transforms=transform)

    train_2007_loader = DataLoader(train_2007_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    print ("loaded 2007 train dataset\n")

    val_2007_dataset = datasets.VOCSegmentation(root='dataset/val', year='2007', image_set='val', download=True, transforms=transform)

    val_2007_loader = DataLoader(val_2007_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    print ("loaded 2007 val dataset\n")

    print ("=======================================\n\n")

    pretrain_model = Yolo_pretrain(conv_only=True, init_weight=True)
    pretrain_model = torch.nn.DataParallel(pretrain_model.features)

    src_state_dict = torch.load(args.pretrain_weight_path)['state_dict']
    dst_state_dict = pretrain_model.state_dict()

    for key in dst_state_dict.keys():
        print ("Loading weight of", key ,"\n")

        dst_state_dict[key] = src_state_dict[key]

    pretrain_model.load_state_dict(dst_state_dict)

    print ("Loaded pretrain weight\n\n")

    print (f"Number of Train images: {len(train_2012_dataset) + len(train_2007_dataset)}")
    print (f"Number of Validation images: {len(val_2012_dataset) + len(val_2007_dataset)}\n")

    print ("======================================\n")


    yolo = Yolo_v1(pretrain_model.features)
    yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layer)

    if is_cuda:
        yolo.cuda()

    # Loss func
    criterion = Yolo_loss(S=S, B=B, C=C)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    log_dir = args.result_path

    logfile = open(os.path.join(log_dir, 'log.txt'), 'w')

    best_val_loss = np.inf

    for epoch in range(num_epochs):
        print (f'\nStarting epoch {epoch + 1} / {num_epochs}')

        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(train_2007_loader):

            update_lr(optimizer, epoch, float(i) / float(len(train_2007_loader) - 1))
            lr = get_lr(optimizer)

            batch_size_iter = img.size(0)

            imgs = Variable(imgs)
            targets = Variable(targets)

            if is_cuda:
                imgs = imgs.cuda()
                targets = targets.cuda()

            # Forawrd
            preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_iter = loss.item()
            total_loss += loss_iter * batch_size_iter
            toatl_batch += batch_size_iter

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isPrint(i):
                print (f'\tEpoch(2007) [{epoch+1}/{num_epochs}]  |  Iter [{i+1}/{len(train_2007_loader)}]  |  LR: {lr:.6f},  Loss: {loss_iter:.4f}, Avg Loss: {total_loss / float(total_batch)}')


        for i, (imgs, targets) in enumerate(train_2012_loader):

            update_lr(optimizer, epoch, float(i) / float(len(train_2007_loader) - 1))
            lr = get_lr(optimizer)

            batch_size_iter = img.size(0)

            imgs = Variable(imgs)
            targets = Variable(targets)

            if is_cuda:
                imgs = imgs.cuda()
                targets = targets.cuda()

            # Forawrd
            preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_iter = loss.item()
            total_loss += loss_iter * batch_size_iter
            toatl_batch += batch_size_iter

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isPrint(i):
                print (f'\tEpoch(2012) [{epoch+1}/{num_epochs}]  |  Iter [{i+1}/{len(train_2007_loader)}]  |  LR: {lr:.6f},  Loss: {loss_iter:.4f}, Avg Loss: {total_loss / float(total_batch)}')

        # val
        yolo.eval()
        val_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(val_2007_loader):
            batch_size_iter = imgs.size(0)
            imgs, target = Variable(imgs), Variable(targets)

            if is_cuda:
                imgs, target = imgs.cuda(), target.cuda()

            with torch.no_grad()
                preds = yolo(imgs)

            loss = criterion(preds, targets)
            loss_iter = loss.item()

            val_loss += loss_iter * batch_size_iter
            total_batch += batch_size_iter


        for i, (imgs, targets) in enumerate(val_2012_loader):
            batch_size_iter = imgs.size(0)
            imgs, target = Variable(imgs), Variable(targets)

            if is_cuda:
                imgs, target = imgs.cuda(), target.cuda()

            with torch.no_grad()
                preds = yolo(imgs)

            loss = criterion(preds, targets)
            loss_iter = loss.item()

            val_loss += loss_iter * batch_size_iter
            total_batch += batch_size_iter

        val_loss /= float(total_batch)

        # Save checkpoint
        logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
        logfile.flush()
        
        torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_latest.pth'))

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))

        print (f'\nEpoch [{epoch+1}/{num_epochs}], Val loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')

   writer.close()
   logfile.close()
