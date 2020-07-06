import torch
import torch.nn as nn

class Yolo_loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_coord=5.0, l_noobj=0.5):
        super(Yolo_loss, self).__init__()

        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj


    # box (x1, y1, w, h)
    # N, M boxes
    def compute_IOU(self, boxes1, boxes2):
        N = boxes1.size(0)
        M = boxes2.size(0)
