import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd import Variable

class Yolo_loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_coord=5.0, l_noobj=0.5):
        super(Yolo_loss, self).__init__()

        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj


    # box (x1, y1, x2, y2)
    # N, M boxes
    def compute_iou(self, boxes1, boxes2):
        N = boxes1.size(0)
        M = boxes2.size(0)

        # (N, 2) -> (N, 1, 2) -> (N, M, 2)
        # (M, 2) -> (1, M, 2) -> (N, M, 2)
        top_left = torch.max(
                boxes1[:, :2].unsqueeze(1).expand(N, M, 2),
                boxes2[:, :2].unsqueeze(0).expand(N, M, 2))

        bottom_right = torch.min(
                boxes1[:, 2:].unsqueeze(1).expand(N, M, 2),
                boxes2[:, 2:].unsqueeze(0).expand(N, M, 2))

        width_height = bottom_right - top_left

        width_height[width_height < 0] = 0 # Clip at 0

        intersection = width_height[:, :, 0] * width_height[:, :, 1]
        # shape: [N, M]

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # shape: [N, ] / [M, ]

        area1 = area1.unsqueeze(1).exapnd_as(intersection)
        area2 = area2.unsqueeze(0).expand_as(intersection)
        # [N, ] -> [N, 1] -> [N, M]
        # [M, ] -> [1, M] -> [N, M]

        union = area1 + area2 - intersection
        iou = intersection / union

        retrun iou


    def forward(self, pred_tensor, target_tensor):
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 5 => x, y, w, h, confidence

        batch_size = pred_tensor.size(0)
        # mask for the cells which contain object
        # shape: [batches, S, S]
        coord_mask = target_tensor[:,:,:,4] > 0
        # mask for the cell which not contain object
        # shape: [batches, S, S]
        noobj_mask = target_tensor[:,:,:,4] == 0

        # [batches, S, S] -> [batches, S, S, N]
        #     (unsqueeez(-1): to make [batches, S, S, 1])
        #     (expand_as(~~): to make [batches, S, S, N]; shape of target_tensor)
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        # Cells which contains object
        # shape: [Number of Object, N]
        coord_pred = pred_tensor[coord_mask].view(-1, N)
        coord_target = target_tensor[coord_mask].view(-1, N)

        # Bounding box info
        # shape: [Number of object, x/y/h/w/c]
        bbox_pred = coord_pred[:, :5*B].view(-1, 5)
        bbox_target = coord_target[:, :5*B].view(-1, 5)

        # Class info
        # shape: [Number of object, class]
        class_pred = coord_pred[:, 5*B:]
        class_target = coord_target[:, 5*B:]


        # Cells which not contains object
        # shape: [Number of no object, N]
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)
        noobj_target = target_tensor[noobj_mask].view(-1, N)

        # Confidence of no objects
        noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0)

        # 4th index: x, y, w, h, "c"
        # b*5 = b th bounding box
        for b in range(B):
            noobj_conf_mask[:,4+b*5]=1

        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')


        # Compute loss for the cells with object
        coord_response_mask = torch.cuda.ByteTensor(bbox_target.size())fill_(0)
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()

        # Choose the predict bbox having the highest IoU for each target bbox
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted B box in i th cell

            # Because target b boxes contained by each cell are identical in this implementation, it is enought to extract the first one.
            target = bbox_target[i].view(-1, N)
            # shape: [B, 5(x, y, w, h, c)]

            pred_xyxy = Variable(torch.FloatTensor(pred.size()))
            target_xyxy = Variable(torch.FloatTensor(target.size()))
            # shape: [B, 5]

            # pred[:, 2] => pred b box's width
            # target_xyxy[:, :2] => b box top left x, y coordinates
            # target_xyxy[:, 2:4] => b box bottom right x, y, coordinates
            pred_xyxy[:, :2] = pred[:, 2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, 2]/float(S) + 0.5 * pred[:, 2:4]
            target_xyxy[:, :2] = target[:, 2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, 2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])

            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1

            # make confidence to IoU of prediction and groud truth
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()

        bbox_target_iou = Variable(bbox_target_iou).cuda()

        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)


        loss_xy = F.mse_loss(bbox_pred_responsep[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        loss = self.l_coord*(loss_xy + loss_wh) + loss_obj + self.l_noobj * loss_noobj + loss_class

        loss = loss / float(batch_size)

        return loss
