import argparse
import math
import os
import torch
import numpy as np
from dataset.PascalVOC import VOCDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from yolo_v1 import Yolo_v1
from loss import Yolo_loss
from pre_train.model import Yolo_pretrain

S = 0
B = 0
C = 0
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

    parser.add_argument('--train-images', type=str,
            default='./dataset/train/images',
            help='train images path')

    parser.add_argument('--train-annotations', type=str,
            default='./dataset/train/annotations',
            help='train annotations path')

    parser.add_argument('--val-images', type=str,
            default='./dataset/val/images',
            help='validation images path')

    parser.add_argument('--val-annotations', type=str,
            default='./dataset/val/annotations',
            help='validation annotations path')

    parser.add_argument("--class-label", type=str,
            default='./dataset/class_list.txt',
            help='class label txt file')

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

    parser.add_argument('--resume', type=bool,
            default=False)

    parser.add_argument('--gpu-num', type=int,
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
        if i % 50 == 0:
            isPrint = True

    return isPrint

def save_checkpoint(state, epoch, log_dir):
    filenames = os.listdir(log_dir)

    for name in filenames:
        if name.startwith('checkpoint'):
            os.remove(name)
            break

    checkpoint_file = log_dir + "/checkpoint_" + str(epoch + 1) + ".pth"
    torch.save(state, checkpoint_file)

if __name__ == "__main__":
    args = set_argument()

    is_cuda = torch.cuda.is_available()

    if args.gpu_num != -1:
        device = torch.device(f'cuda:{args.gpu_num}' if is_cuda else 'cpu')
        torch.cuda.set_device(device)

    train_dataset = VOCDataset(is_train=True, image_dir=args.train_images,
            annotation_dir=args.train_annotations, label_txt=args.class_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=8)

    val_dataset = VOCDataset(is_train=False, image_dir=args.val_images,
            annotation_dir=args.val_annotations, label_txt=args.class_label)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=8)

    print ("Loaded dataset!")
    print (f"\tNumber of training images: {len(train_dataset)}")
    print (f"\tNumber of validation images: {len(val_dataset)}\n")


    print ("=======================================\n\n")

    pretrain_model = Yolo_pretrain(conv_only=True, init_weight=True)
    pretrain_model.features = torch.nn.DataParallel(pretrain_model.features)

    src_state_dict = torch.load(args.pretrain_weight_path)['state_dict']
    dst_state_dict = pretrain_model.state_dict()

    for key in dst_state_dict.keys():
        dst_state_dict[key] = src_state_dict[key]

    pretrain_model.load_state_dict(dst_state_dict)

    print ("Loaded pretrain weight\n\n")

    print (f"Number of Train images: {len(train_dataset)}")
    print (f"Number of Validation images: {len(val_dataset)}\n")

    print ("======================================\n")


    yolo = Yolo_v1(pretrain_model.features)
    yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layer)

    if is_cuda:
        if args.gpu_num != -1:
            yolo.cuda(args.gpu_num)
        else:
            yolo.cuda()

    # Loss func
    criterion = Yolo_loss()
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    log_dir = args.result_path

    logfile = open(os.path.join(log_dir, 'log.txt'), 'w')

    best_val_loss = np.inf

    start_epoch = 0

    # Load checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] - 1
            best_val_loss = checkpoint['best_val_loss']
            yolo.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print ("Loaded checkpoint!")
            print (f"Start epoch from {start_epoch + 1}")
            print (f"Lastest best val loss: {best_val_loss}\n")

    for epoch in range(start_epoch, args.epochs):
        print (f'\nStarting epoch {epoch + 1} / {args.epochs}')

        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(train_loader):

            update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
            lr = get_lr(optimizer)

            batch_size_iter = imgs.size(0)

            imgs = Variable(imgs)
            targets = Variable(targets)

            if is_cuda:
                if args.gpu_num != -1:
                    imgs = imgs.cuda(args.gpu_num, non_blocking=True)
                    targets = targets.cuda(args.gpu_num, non_blocking=True)

                else:
                    imgs, targets = imgs.cuda(), targets.cuda()

            # Forawrd
            preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_iter = loss.item()
            total_loss += loss_iter * batch_size_iter
            total_batch += batch_size_iter

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isPrint(i, len(train_loader)):
                print (f'\tEpoch [{epoch+1}/{args.epochs}]  |  Iter [{i+1}/{len(train_loader)}]  |  LR: {lr:.6f}')
                print (f'\t    Loss: {loss_iter:.4f}, Avg Loss: {(total_loss / float(total_batch)):.6f}')


        # val
        yolo.eval()
        val_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(val_loader):
            batch_size_iter = imgs.size(0)
            imgs, target = Variable(imgs), Variable(targets)

            if is_cuda:
                imgs, target = imgs.cuda(args.gpu_num), target.cuda(args.gpu_num)

            with torch.no_grad():
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
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': yolo.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict()},
            epoch+1, log_dir)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))

        print (f'\nEpoch [{epoch+1}/{args.epochs}], Val loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')

    writer.close()
    logfile.close()
