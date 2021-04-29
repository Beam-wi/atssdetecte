import torch
from torch import nn
import torch.nn.functional as F
from bbox.bbox import box_decode


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        # print(target.sum())
        eps = 1e-6
        pt = torch.sigmoid(_input)
        #pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt + eps) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + eps)
        # print(loss)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class ATSSLoss(nn.Module):
    def __init__(self, num_classes):
        super(ATSSLoss, self).__init__()

        self.cls_loss_func = BCEFocalLoss(reduction='sum')
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum") 
        self.num_classes = num_classes

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = box_decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = box_decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


    def forward(self, preds, targets, anchors):
        logits, bbox_reg, centerness = preds

        batch_size = logits.shape[0]
        # print('pred', logits.shape, bbox_reg.shape)

        cls_targets, reg_targets, centerness_targets = targets
        # print('targets', cls_targets.shape, reg_targets.shape)

        logits_flatten = logits.reshape(-1, self.num_classes)
        bbox_reg_flatten = bbox_reg.reshape(-1, 4)
        centerness_flatten = centerness.reshape(-1)

        cls_targets_flatten = cls_targets.reshape(-1)
        reg_targets_flatten = reg_targets.reshape(-1, 4)
        centerness_targets_flatten = centerness_targets.reshape(-1)

        # print('cls_targets_flatten', cls_targets_flatten)
        is_positive = (cls_targets_flatten > -1)
        pos_inds = torch.nonzero(is_positive, as_tuple=False).squeeze(1)
        positive_num = pos_inds.numel()

        # check predict
        # idx = pos_inds[:min(positive_num, 10)]
        # print('before', pos_inds, pos_inds.numel())
        # print('before backward predict cls score', logits_flatten[idx, :].sigmoid())
        # print('positive_num', positive_num, pos_inds, pos_inds.shape)
        # print('is_positive', is_positive, is_positive.shape)

        cls_targets_flatten[is_positive == False] = 0
        if self.num_classes > 1:
            cls_targets_flatten = F.one_hot(cls_targets_flatten, self.num_classes).squeeze(dim=1)
            cls_targets_flatten[is_positive == False, :] = 0
        elif self.num_classes == 1:
            cls_targets_flatten[is_positive == True] = 1
            cls_targets_flatten = cls_targets_flatten.unsqueeze(1)

        # print('cls_targets', cls_targets[:, 0].sum(), cls_targets[:, 1].sum())
        # print(logits_flatten.shape, cls_targets_flatten.shape)
        cls_loss = self.cls_loss_func(logits_flatten, cls_targets_flatten) / max(positive_num, 1.0)
        # cls_loss = self.cls_loss_func(logits_flatten, cls_targets_flatten) / logits_flatten.shape[0]

        bbox_reg_flatten = bbox_reg_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        centerness_targets_flatten = centerness_targets_flatten[pos_inds]
        anchors = torch.cat(anchors).expand(batch_size, -1, 4).reshape(-1, 4)[pos_inds]
        # print(centerness_flatten[:10], centerness_targets_flatten[:10])
        # print(anchors)
        # print('positive_num', positive_num, centerness_flatten.shape)
        if positive_num > 0:
            # centerness loss
            # print(bbox_reg_flatten.type(), reg_targets_flatten.type())
            reg_loss = self.GIoULoss(bbox_reg_flatten, reg_targets_flatten, anchors, weight=centerness_targets_flatten) / positive_num
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets_flatten) / positive_num
        else:
            print('no positive samples')
            reg_loss = bbox_reg_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return {"loss_cls": cls_loss, "loss_reg": reg_loss, "loss_centerness": centerness_loss, 'is_positive':is_positive}

