# In Paper,
# learning_rate = 0.001
# num_epochs = 135
# batch_size = 64
# momentum = 0.9
# decay = 0.0005

import argparse
from model.model import Yolo_v1

S = 0
B = 0
C = 0
weight_path = ""
test_path = ""
train_path = ""
lr = 0.0

def set_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', type=int,
            default=7,
            help="grid count, deafult = 7")

    parser.add_argument('-b', type=int,
            default=2,
            help="number of boundary box, deafult = 2")

    parser.add_argument('-w', '--weight-path', type=str,
            default='./weight/',
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

    args = parser.parse_args()

    S = args.s
    B = args.b
    
    weight_path = args.weight_path
    test_path = args.test_data
    train_path = args.train_data

if __name__ == "__main__":
    set_argument()

    yolo_net = Yolo_v1(S, B, C)
