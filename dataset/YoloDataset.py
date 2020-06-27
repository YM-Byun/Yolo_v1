import torch
import torch.utils.data as data

class YoloDataset(data.Dataset):
    image_size =  448

    def __init__(self, root, list_file, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            # Token[0] : Image file name
            # others: coordinates
            token = line.strip().split()
            self.fnames.append(token[0])
            num_boxes = (len(token) - 1)

            box = []
            label = []

            for i in range(num_boxes):
                x = float(token[1 + 5*i])
                y = float(token[2 + 5*i])
                h = float(token[3 + 5*i])
                w = float(token[4 + 5*i])
                c = token[5 + 5*i]

                box.append([x, y, h, w])
                labels.append(int(c) + 1)

            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples
