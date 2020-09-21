# Yolo v1 re-implementation


## Pre-training
### 1. pretraing for 20 convolution layers using ImageNet 2012 datasets.
 - Top 5 87% acc

## Training
### 1. Using PASCAL 2007, 2012 datasets.
 - 15496 images for training (70%)
 - 6640 images for validation (30%)
### 2. 135 epochs, batch-size 64, momentum 0.9, decay 0.0005
### 3. Different learning rate for epochs
### 4. extensive data augmentation

## Inference
![result](https://user-images.githubusercontent.com/29909314/93767250-b0af0780-fc52-11ea-9134-5f20d93a4c75.png)
