# In Paper,
# learning_rate = 0.001
# num_epochs = 135
# batch_size = 64
# momentum = 0.9
# decay = 0.0005

import argparse
import math
import torch
import numpy as np
from model.model import Yolo_v1

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

    parser.add_argument('-w', '--weight-path', type=str,
            help="pre-trained weight path")

    parser.add_argument('--train-data', type=str,
            default='./train_data',
            help='train dataset path')

    parser.add_argument('--test-data', type=str,
            default='./test_data',
            help='test dataset path')

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

    S = args.s
    B = args.b

    num_epochs = args.epochs
    
    weight_path = args.weight_path
    test_path = args.test_data
    train_path = args.train_data

    decay = args.weight_decay
    momentum = args.momentum
    batch_size = args.batch_size

    learning_rate = args.lr

def get_lr(epoch):
    lr = 0

    if epoch == 0:
        lr = 0.001

    elif epoch < 75:
        lr = 0.01

    elif epoch < 105:
        lr = 0.001

    elif epoch < 135:
        lr = 0.0001

    else:
        return

    return lr

if __name__ == "__main__":
    set_argument()

    is_cuda = torch.cuda.is_available()

    yolo_net = Yolo_v1(S, B, C)

    if is_cuda:
        yolo_net.cuda()

    # load pre-trained weight
    if weight_path != "":
        yolo_net.load_state_dict(torch.load(weight_path))

    #criterion = yolo_loss()
    optimizer = torch.optim.SGD(yolo_net.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=decay)

    yolo_net.train()

    # Dataset Load..
    # End

    for epoch in range(num_epochs):
        print ("--------------------------------------\n")
        print (f"Starting epoch {epoch + 1} / {num_epochs}")

        learning_rate = get_lr(epoch)
        print (f"Learning Rate: {learning_rate}")

        yolo_net.train()

        if epoch == 0:
            learning_rate = np.linspace(learning_rate, 0.01, num=10)

        # for i, (imgs, targets) in enumerate(train_loader):
        for i in range(10):
            if epoch == 0:
                print (f"Learning rate in irst epoch: {learning_rate[i]}")

        print (" ")
