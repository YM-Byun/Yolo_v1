import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Yolo_v1(nn.Module):
    def __init__(self, S, B, C):
        super(Yolo_v1, self).__init__()
        
        self.S = S
        self.B = B
        self.C = C
        
        # * -> unpacking list
        self.layer1 = nn.Sequential(
            *self.make_conv_layer(3, 64, 7, 2, 3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )       
        
        self.layer2 = nn.Sequential(
            *self.make_conv_layer(64, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            *self.make_conv_layer(192, 128, 1, 1, 0),
            *self.make_conv_layer(128, 256, 3, 1, 1),
            *self.make_conv_layer(256, 256, 1, 1, 0),
            *self.make_conv_layer(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer4 = nn.Sequential(
            *self.make_conv_layer(512, 256, 1, 1, 0),
            *self.make_conv_layer(256, 512, 3, 1, 1),
            *self.make_conv_layer(512, 256, 1, 1, 0),
            *self.make_conv_layer(256, 512, 3, 1, 1),
            *self.make_conv_layer(512, 256, 1, 1, 0),
            *self.make_conv_layer(256, 512, 3, 1, 1),
            *self.make_conv_layer(512, 256, 1, 1, 0),
            *self.make_conv_layer(256, 512, 3, 1, 1),
            
            *self.make_conv_layer(512, 512, 1, 1, 0),
            *self.make_conv_layer(512, 1024, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            *self.make_conv_layer(1024, 512, 1, 1, 0),
            *self.make_conv_layer(512, 1024, 3, 1, 1),
            *self.make_conv_layer(1024, 512, 1, 1, 0),
            *self.make_conv_layer(512, 1024, 3, 1, 1),
            
            *self.make_conv_layer(1024, 1024, 3, 1, 1),
            *self.make_conv_layer(1024, 1024, 3, 2, 1)
        )
        
        self.layer6 = nn.Sequential(
            *self.make_conv_layer(1024, 1024, 3, 1, 1),
            *self.make_conv_layer(1024, 1024, 3, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )
        
    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = []
        layer.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=stride,
                      padding=padding))
        
        layer.append(nn.LeakyReLU())
        
        return layer
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = x.view(-1, 7*7*1024)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        
        return x


if __name__ == '__main__':
    dummy_data = torch.rand((10, 3, 448, 448))
    yolo = Yolo_v1(7, 2, 20)
    x = yolo(dummy_data)

    print ("Yolo V1 network")
    print (yolo)

    print ("-------------------------")
    print (f'Result: {x.shape}')
