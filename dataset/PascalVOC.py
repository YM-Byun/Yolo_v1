import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import random
import numpy as np
import cv2

class VOCDataset(Dataset):

    def __init__(self, is_train, image_dir, annotation_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.is_train = is_train
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.paths, self.boxes, self.labels = [], [], []

        file_list = os.listdir(image_dir)

        # Load class_label info
        label_info = self.get_class_label_dict(label_txt)

        for image in file_list:

            fname = image
            path = os.path.join(image_dir, fname)
            self.paths.append(path)

            data = self.parse_annotations(os.path.join(annotation_dir,
                            fname.replace(".jpg", ".xml")))

            box, label = [], []
            for i in range(data['num_boxes']):
                if data['c'][i] not in label_info:
                    continue

                x1 = data['x1'][i]
                y1 = data['y1'][i]
                x2 = data['x2'][i]
                y2 = data['y2'][i]
                c  = label_info[data['c'][i]]
                box.append([x1, y1, x2, y2])
                label.append(c)

            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.paths)

    def get_class_label_dict(self, label_txt):
        lines = []

        label_info = {}

        with open(label_txt, 'r') as label_file:
            lines = label_file.readlines()

        for line in lines:
            line = line.replace('\n', '')
            
            token = line.split('|')

            label_info[token[0]] = int(token[1])

        return label_info


    def parse_annotations(self, annotation_xml):
        xml_code = []
        with open(annotation_xml, 'r') as xml:
            xml_code = xml.readlines()

        is_object = False

        c = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for line in xml_code:
            line = line.replace('\n', '')

            if '</object>' in line:
                is_object = False

            if is_object:
                if "<name>" in line:
                    name = self.get_info_from_tag(line, '<name>')
                    c.append(name)

                if "<xmin>" in line:
                    x_coord = self.get_info_from_tag(line, '<xmin>')
                    x1.append(float(x_coord))

                if "<ymin>" in line:
                    y_coord = self.get_info_from_tag(line, '<ymin>')
                    y1.append(float(y_coord))

                if "<xmax>" in line:
                    x_coord = self.get_info_from_tag(line, '<xmax>')
                    x2.append(float(x_coord))

                if "<ymax>" in line:
                    y_coord = self.get_info_from_tag(line, '<ymax>')
                    y2.append(float(y_coord))
                    
            if '<object>' in line:
                is_object = True

        return {"num_boxes": len(c), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'c': c}

    def get_info_from_tag(self, line, open_tag):
        close_tag = '</' + open_tag[1:]
        info = line.strip().replace(open_tag, '').replace(close_tag, '')

        return info

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        boxes = self.boxes[idx].clone() # [n, 4]
        labels = self.labels[idx].clone() # [n,]

        if self.is_train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)

            img = self.random_blur(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)

            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # For debug.
        debug_dir = 'tmp/voc_tta'
        os.makedirs(debug_dir, exist_ok=True)
        img_show = img.copy()
        box_show = boxes.numpy().reshape(-1)
        n = len(box_show) // 4
        for b in range(n):
            pt1 = (int(box_show[4*b + 0]), int(box_show[4*b + 1]))
            pt2 = (int(box_show[4*b + 2]), int(box_show[4*b + 3]))
            cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0,255,0), thickness=1)
        cv2.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        target = self.encode(boxes, labels) # [S, S, 5 x B + C]

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # assuming the model is pretrained with RGB images.
        img = (img - self.mean) / 255.0 # normalize from -1.0 to 1.0.
        img = self.to_tensor(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size # x & y of the box on the cell, normalized from 0.0 to 1.0.

            # TBM, remove redundant dimensions from target tensor.
            # To remove these, loss implementation also has to be modified.
            for k in range(B):
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0
            target[j, i, 5*B + label] = 1.0

        return target

    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes

        h, w, _ = img.shape

        img = np.fliplr(img)

        x1, x2 = boxes[:, 0], boxes[:, 2]
        x1_new = w - x2
        x2_new = w - x1
        boxes[:, 0], boxes[:, 2] = x1_new, x2_new

        return img, boxes

    def random_scale(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes

        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(int(w * scale), h), interpolation=cv2.INTER_LINEAR)

        scale_tensor = torch.FloatTensor([[scale, 1.0, scale, 1.0]]).expand_as(boxes)
        boxes = boxes * scale_tensor

        return img, boxes

    def random_blur(self, bgr):
        if random.random() < 0.5:
            return bgr

        ksize = random.choice([2, 3, 4, 5])
        bgr = cv2.blur(bgr, (ksize, ksize))
        return bgr

    def random_brightness(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_hue(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.8, 1.2)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_shift(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h, w, c = img.shape
        img_out = np.zeros((h, w, c), dtype=img.dtype)
        mean_bgr = self.mean[::-1]
        img_out[:, :] = mean_bgr

        dx = random.uniform(-w*0.2, w*0.2)
        dy = random.uniform(-h*0.2, h*0.2)
        dx, dy = int(dx), int(dy)

        if dx >= 0 and dy >= 0:
            img_out[dy:, dx:] = img[:h-dy, :w-dx]
        elif dx >= 0 and dy < 0:
            img_out[:h+dy, dx:] = img[-dy:, :w-dx]
        elif dx < 0 and dy >= 0:
            img_out[dy:, :w+dx] = img[:h-dy, -dx:]
        elif dx < 0 and dy < 0:
            img_out[:h+dy, :w+dx] = img[-dy:, -dx:]

        center = center + torch.FloatTensor([[dx, dy]]).expand_as(center) # [n, 2]
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) # [n,]
        mask = (mask_x & mask_y).view(-1, 1) # [n, 1], mask for the boxes within the image after shift.

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) # [m, 4]
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out) # [m, 4]

        boxes_out = boxes_out + shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]

        return img_out, boxes_out, labels_out

    def random_crop(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h_orig, w_orig, _ = img.shape
        h = random.uniform(0.6 * h_orig, h_orig)
        w = random.uniform(0.6 * w_orig, w_orig)
        y = random.uniform(0, h_orig - h)
        x = random.uniform(0, w_orig - w)
        h, w, x, y = int(h), int(w), int(x), int(y)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center) # [n, 2]
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) # [n,]
        mask = (mask_x & mask_y).view(-1, 1) # [n, 1], mask for the boxes within the image after crop.

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) # [m, 4]
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_out) # [m, 4]

        boxes_out = boxes_out - shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]
        img_out = img[y:y+h, x:x+w, :]

        return img_out, boxes_out, labels_out


def test():
    from torch.utils.data import DataLoader

    image_dir = 'data/VOC_allimgs/'
    label_txt = ['data/voc2007.txt', 'data/voc2012.txt']

    dataset = VOCDataset(True, image_dir, label_txt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(100):
        img, target = next(data_iter)
        print(img.size(), target.size())

if __name__ == '__main__':
    test()
