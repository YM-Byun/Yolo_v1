import argparse
import math
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

from yolo_v1 import Yolo_v1
from loss import Yolo_loss
from pre_train.model import Yolo_pretrain

S = 0
B = 0
C = 0
weight_path = ""
test_path = ""
train_path = ""
decay = 0.0005
momentum = 0.9
batch_size = 64
num_epochs = 135
learning_rate = 0.0

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
            default='pre-train/weight/model_best.pth.tar',
            help="pre-trained weight path")

    parser.add_argument('--train-path', type=str,
            default='./dataset/train',
            help='train dataset path')

    parser.add_argument('--val-path', type=str,
            default='./dataset/val',
            help='validation dataset path')

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = set_argument()

    is_cuda = torch.cuda.is_available()

    normalize = transforms.Normalize(mean=[0.485, 0.485, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.RandomHorizontalFlip()])

    train_2012_dataset = datasets.VOCSegmentation(root='dataset/train', year='2012', image_set='train', download=True, transforms=transform)

    print ("loaded 2012 train dataset\n")

    val_2012_dataset = datasets.VOCSegmentation(root='dataset/val', year='2012', image_set='val', download=True, transforms=transform)

    print ("loaded 2012 val dataset\n")

    train_2007_dataset = datasets.VOCSegmentation(root='dataset/train', year='2007', image_set='train', download=True, transforms=transform)

    print ("loaded 2007 train dataset\n")

    val_2007_dataset = datasets.VOCSegmentation(root='dataset/val', year='2007', image_set='val', download=True, transforms=transform)

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

    print ("Loaded pretrain weight\n")

    print ("======================================\n\n")


    yolo = Yolo_v1(pretrain_model.features)
    yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layer)

    if is_cuda:
        yolo.cuda()

    # Loss func
    criterion = Yolo_loss(S=S, B=B, C=C)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # Load Pasacl-VOC daaset
