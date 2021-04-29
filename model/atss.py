import torch
from torch import nn
from torch.nn import functional as F
import math

from .resnet import resnet18
from .neck import PyramidFeatures
from .head import HeadModel
from .anchor import prepare_anchors
from bbox.bbox import bbox_overlaps_torch as bbox_overlaps
from bbox.bbox import box_encode

INF = 100000000

def prepare_targets(annos, anchors, topk=9, device=torch.device("cuda:0")):
    anchor_nums = [anchor.shape[0] for anchor in anchors]
    anchors = torch.cat(anchors).to(device)
    total_anchors_num = anchors.shape[0]

    gt_boxes = annos['bboxes'].float().to(device)
    obj_nums = annos['obj_num'].to(device)
    obj_clses = annos['cls'].to(device)
    batch_size = gt_boxes.shape[0]

    # print('prepare_targets', anchors.device, gt_boxes.device)

    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2.0
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2.0
    anchor_points = torch.stack((anchors_cx, anchors_cy), dim=1)

    cls_targets = []
    reg_targets = []
    ctness_targets = []
    for i in range(batch_size):
        # print('gt number:', obj_nums[i])
        num_gt = obj_nums[i]
        if num_gt == 0:
            cls_targets.append(torch.full((anchors.shape[0],), -1, dtype=torch.int).to(device))
            reg_targets.append(torch.zeros_like(anchors).to(device))
            ctness_targets.append(torch.zeros((anchors.shape[0],), dtype=torch.float).to(device))
            continue

        bboxes_per_img = gt_boxes[i][:num_gt]
        labels_per_im = obj_clses[i][:num_gt]
        # print('labels_per_im', labels_per_im)

        gt_cx = (bboxes_per_img[:, 2] + bboxes_per_img[:, 0]) / 2.0
        gt_cy = (bboxes_per_img[:, 3] + bboxes_per_img[:, 1]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        # print(gt_points)
        # print(gt_points.shape, anchor_points.shape)
        distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        # print('distance', distances.shape)

        candidate_idxs = []
        star_idx = 0
        for anchor_num_per_level in anchor_nums:
            end_idx = star_idx + anchor_num_per_level
            distances_per_level = distances[star_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
            # print('topk_idxs_per_level', topk_idxs_per_level.shape)
            candidate_idxs.append(topk_idxs_per_level + star_idx)
            star_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        # print('candidate_idxs', candidate_idxs)

        ious = bbox_overlaps(anchors.float(), bboxes_per_img.float())
        # print('ious', ious.shape, candidate_idxs.shape)
        candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
        # print('candidate_ious', candidate_ious)

        iou_mean_per_gt = candidate_ious.mean(0)
        iou_std_per_gt = candidate_ious.std(0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]
        # print('is_pos', is_pos.shape)

        # Limiting the final positive samples’ center to object
        for ng in range(num_gt):
            candidate_idxs[:, ng] += ng * total_anchors_num
        e_anchors_cx = anchors_cx.view(1, -1).expand(num_gt, total_anchors_num).contiguous().view(-1)
        e_anchors_cy = anchors_cy.view(1, -1).expand(num_gt, total_anchors_num).contiguous().view(-1)

        candidate_idxs = candidate_idxs.view(-1)
        l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_img[:, 0]
        t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_img[:, 1]
        r = bboxes_per_img[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
        b = bboxes_per_img[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
        # print('l,t,r,b', l.shape)
        is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
        # print('is_in_gts', is_in_gts.shape)
        is_pos = is_pos & is_in_gts
        # print('is_pos & is_in_gts', is_pos.shape, is_pos)

        # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gt, -1).t()
        # print('ious_inf', ious_inf[0], ious_inf.shape)

        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
        # print('anchors_to_gt_values', anchors_to_gt_values, anchors_to_gt_indexs, anchors_to_gt_values.shape)
        # print('max', ious_inf[0, anchors_to_gt_indexs[0]])
        # print(labels_per_im)
        cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
        # print('cls_labels_per_im', cls_labels_per_im, len(cls_labels_per_im))
        cls_labels_per_im[anchors_to_gt_values == -INF] = -1
        # print('cls_labels_per_im', cls_labels_per_im)
        matched_gts = bboxes_per_img[anchors_to_gt_indexs]
        # print('matched_gts', bboxes_per_img.shape, matched_gts.shape, cls_labels_per_im.shape)
    
        reg_targets_per_im = box_encode(matched_gts, anchors)
        cls_targets.append(cls_labels_per_im)
        reg_targets.append(reg_targets_per_im)
        # print('res', cls_labels_per_im.shape, reg_targets_per_im.shape)

        # centerness
        l = anchors_cx - matched_gts[:, 0]
        t = anchors_cy - matched_gts[:, 1]
        r = matched_gts[:, 2] - anchors_cx
        b = matched_gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1).abs()
        top_bottom = torch.stack([t, b], dim=1).abs()
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        ctness_targets.append(centerness)

    cls_targets = torch.stack(cls_targets, dim=0)
    reg_targets = torch.stack(reg_targets, dim=0)
    ctness_targets = torch.stack(ctness_targets, dim=0)

    return cls_targets, reg_targets, ctness_targets


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ATSSModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super(ATSSModel, self).__init__()

        anchor_strides = [8, 16, 32, 64, 128]
        anchor_scale = [8]

        self.num_classes = num_classes

        self.backbone = resnet18()

        inputs = torch.rand(1, *input_size, requires_grad=False)
        fpn = self.backbone(inputs)
        fpn_sizes = [fpn[0].size(1), fpn[1].size(1), fpn[2].size(1)]
        feature_size = 256

        self.neck = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], feature_size=feature_size)


        self.head = HeadModel(feature_size, num_anchors=1, feature_size=256, num_classes=num_classes)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        neck = self.neck(fpn)
        self.neck_shapes = [[each.size(2), each.size(3)] for each in neck]
        self.anchors = prepare_anchors(self.neck_shapes, anchor_strides, anchor_scale)
        # self.anchors = torch.Tensor(self.anchors)

        for modules in [self.neck, self.head]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.xavier_uniform_(l.weight)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.head.classificationModel.classifier.bias, bias_value)

    def forward(self, x):
        x = self.backbone(x)
        feat_level_maps = self.neck(x)

        logits = []
        bbox_reg = []
        centerness = []

        for i, feat_maps in enumerate(feat_level_maps):
            obj_cls, reg, ctness = self.head(feat_maps)

            bbox_pred = self.scales[i](reg)

            logits.append(obj_cls)
            bbox_reg.append(bbox_pred)
            centerness.append(ctness)
        
        logits = torch.cat(logits, dim=1)
        bbox_reg = torch.cat(bbox_reg, dim=1)
        centerness = torch.cat(centerness, dim=1)

        return logits, bbox_reg, centerness


