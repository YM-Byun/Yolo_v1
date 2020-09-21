import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pretrain.model import Yolo_pretrain

class Yolo_v1(nn.Module):
    def __init__(self, features, S=7, B=2, C=20):
        super(Yolo_v1, self).__init__()
        
        self.S = S
        self.B = B
        self.C = C
        self.features = features
        
        # * -> unpacking list
        self.layer1 = nn.Sequential(
            *self.make_conv_layer(1024, 1024, 3, 1, 1),
            *self.make_conv_layer(1024, 1024, 3, 2, 1)
        )
        
        self.layer2 = nn.Sequential(
            *self.make_conv_layer(1024, 1024, 3, 1, 1),
            *self.make_conv_layer(1024, 1024, 3, 1, 1)
        )

        self.conv_layers = nn.Sequential(
            self.layer1,
            self.layer2)
        
        self.fc = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
            nn.Sigmoid()
        )
        
    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = []
        layer.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=stride,
                      padding=padding))

        layer.append(nn.BatchNorm2d(out_channels))
        
        layer.append(nn.LeakyReLU())
        
        return layer
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)

        return x


if __name__ == '__main__':
    pretrain_model = Yolo_pretrain(conv_only=True, init_weight=True)
    pretrain_model.features = pretrain_model.features.cuda()

    src_state_dict = torch.load('./pretrain/weight/best.pth')['state_dict']
    dst_state_dict = pretrain_model.state_dict()

    for key in dst_state_dict.keys():
        dst_state_dict[key] = src_state_dict[key]

    pretrain_model.load_state_dict(dst_state_dict)

    dummy_data = torch.rand((10, 3, 448, 448))
    yolo = Yolo_v1(pretrain_model.features, 7, 2, 20).cuda()

    dummy_data = dummy_data.cuda()
    x = yolo(dummy_data)

    print ("Yolo V1 network")
    print (yolo)

    print ("-------------------------")
    print (f'Result: {x.shape}')
