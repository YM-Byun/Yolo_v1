import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util_layers import Squeeze

class Yolo_pretrain(nn.Module):
    def __init__(self, conv_only=False, init_weight=True):
        super(Yolo_pretrain, self).__init__()

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
            *self.make_conv_layer(512, 1024, 3, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 980)
        )

        self.conv_only = conv_only

        if init_weight:
            self.init_weights()
        
    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = []
        layer.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=stride,
                      padding=padding))

        layer.append(nn.BatchNorm2d(out_channels))
        
        layer.append(nn.LeakyReLU(0.1, inplace=True))
        
        return layer
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
       
        if not self.conv_only:
            x = self.fc(x)
       
        return x

    def init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                # He init
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    yolo = Yolo_pretrain()

    print ("Yolo V1 network")
    print (yolo)
