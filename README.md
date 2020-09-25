# Yolo v1 re-implementation


## Pre-training
### 1. pretraing for 20 convolution layers using ImageNet 2012 datasets.
 - until Top 5 87% acc

## Training
### 1. Using PASCAL 2007, 2012 datasets.
 - 15496 images for training (70%)
 - 6640 images for validation (30%)
### 2. 135 epochs, batch-size 64, momentum 0.9, decay 0.0005
### 3. Different learning rate for epochs
### 4. extensive data augmentation

## Experiment
mAP 51.7 (Implement) <br>
    63.4 (In paper)

Class | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
AP | 74.8 | 51.7 | 56.9 | 36.6 | 16.0 | 68.0 | 46.3 | 85.1 | 26.7 | 51.7

Class | dining table | dog | horse | motorbike | person | potted plant | sheep | sofa | train | tv&monitor
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
AP | 37.2 | 77.9 | 52.6 | 54.8 | 61.6| 19.1 | 54.1 | 43.8 | 75.4 | 43.7



## Test Images
![horse_and_person](https://user-images.githubusercontent.com/29909314/93767322-d3412080-fc52-11ea-8c9f-0fffd2274e2d.png)
![result](https://user-images.githubusercontent.com/29909314/93767250-b0af0780-fc52-11ea-9134-5f20d93a4c75.png)
