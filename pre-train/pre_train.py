# In Paper,
# learning_rate = 0.001
# num_epochs = 135
# batch_size = 64
# momentum = 0.9
# decay = 0.0005
print ("Importing pytorch now...\n")

import argparse
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import time
from model import Yolo_pretrain
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

weight_path = ""
dataset_path = ""
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
    
    parser.add_argument('-w', '--weight-path', type=str,
            help="pre-trained weight path")

    parser.add_argument('--dataset-path', type=str,
            default='./dataset',
            help='dataset path')

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

    num_epochs = args.epochs
    
    weight_path = args.weight_path
    dataset_path = args.dataset_path

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

def run_epoch(device, model, train_data, test_data, optimizer, criterion, epoch_num):
    print ("Start train dataset")

    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    
    start_time = time.time()

    cnt = 1
    for img_i, label_i in train_data:
        if cnt < 1000:
            if (cnt % 100 == 0):
                print (f"Train (epoch {epoch_num}) | current {cnt} / {len(train_data)}")
        elif cnt >= 1000:
            if (cnt % 500 == 0):
                print (f"Train (epoch {epoch_num}) | current {cnt} / {len(train_data)}")
        elif cnt == len(test_data) - 1:
                print (f"Train (epoch {epoch_num}) | current {cnt} / {len(train_data)}")
        img_i, label_i = img_i.to(device), label_i.to(device)

        optimizer.zero_grad()

        # Forward
        label_predicted = model.forward(img_i)

        # Loss computatoin
        loss = criterion(label_predicted, label_i.view(-1))

        # Backward
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        pred = label_predicted.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(label_i.data.view_as(pred)).sum()

        
        cnt += 1

    total_test_loss = 0
    cnt = 1

    print ("Start test dataset")
    for img_j, label_j in test_data:

        if cnt < 1000:
            if (cnt % 100 == 0):
                print (f"Test (epoch {epoch_num}) | current {cnt} / {len(test_data)}")

        elif cnt >= 1000:
            if (cnt % 500 == 0):
                print (f"Test (epoch {epoch_num}) | current {cnt} / {len(test_data)}")

        elif cnt == len(test_data) - 1:
                print (f"Test (epoch {epoch_num}) | current {cnt} / {len(test_data)}")

        img_j, label_j = img_j.to(device), label_j.to(device)

        with torch.autograd.no_grad():
            label_predicted = model.forward(img_j)
            val_loss += criterion(label_predicted, label_j.view(-1))
        cnt += 1

    end_time = time.time()

    return train_loss, val_loss, (train_acc / float(len(train_data)) * 100.0),(end_time - start_time)

if __name__ == "__main__":
    set_argument()

    yolo_net = Yolo_pretrain()
    device = torch.device('cpu')

    if torch.cuda.is_available():
        yolo_net.cuda()
        device = torch.device('cuda')

        print ("Using CUDA")


    # load pre-trained weight
    if weight_path != "":
        yolo_net.load_state_dict(torch.load(weight_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(yolo_net.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=decay)

    yolo_net.train()

    # Dataset Load
    print("Load Imagenet Dataset ...")
    transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = ImageFolder(root='./dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = ImageFolder(root='./dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    print ("ImageNet Dataset Load Complete!")
    print (f"Train labels: {len(train_dataset.classes)}")
    print (f"Train data amount: {len(train_dataset)}\n")
    print (f"Test labels: {len(test_dataset.classes)}")
    print (f"Test data amount: {len(test_dataset)}\n\n")

    records = []
    for i, epoch in enumerate(range(num_epochs)):
        print ("-----------------------------------------")
        print (f"Epoch {epoch + 1} / {num_epochs}")
        train_loss, val_loss, train_acc, response = run_epoch(device, yolo_net, train_loader, test_loader, optimizer, criterion, i+1)
        print (f"\tTrain loss: {train_loss:.3f}")
        print (f"\tValid loss: {val_loss:.3f}")
        print (f"\tTrain acc: {train_acc:.3f}")
        print (f"\tResponse time: {response}")

        recode = [train_loss, val_loss, train_acc, response]
        records.append(recode)

        if i % 10 == 0:
            torch.save(yolo_net.state_dict(), "weight.pth")

    for i, r in enumerate(records):
        print ("\n-----------------------------------------")
        print (f" For Epoch {i + 1} / {len(records)}")
        print (f"\tTrain Loss: {r[0]:.3f}")
        print (f"\tValid Loss: {r[1]:.3f}")
        print (f"\tTrain acc: {r[2]:.3f}")
        print (f"\tResponse time: {r[3]}")

    torch.save(yolo_net.state_dict(), "weight.pth")

    print ("Save model weight")
