import torch
from torch import nn


class HeadModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, num_classes=80):
        super(HeadModel, self).__init__()

        self.regressionModel = RegressionModel(num_features_in,
                                               num_anchors=num_anchors,
                                               feature_size=feature_size)
        self.classificationModel = ClassificationModel(num_features_in,
                                                       num_anchors=num_anchors,
                                                       num_classes=num_classes,
                                                       feature_size=feature_size)

    def forward(self, x):
        reg, centerness = self.regressionModel(x)
        obj_cls = self.classificationModel(x)
        return obj_cls, reg, centerness


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output_reg = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        self.output_centerness = nn.Conv2d(feature_size, num_anchors * 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        reg = self.output_reg(out)
        centerness = self.output_centerness(out)

        # out is B x C x H x W, with C = 4*num_anchors
        reg = reg.permute(0, 2, 3, 1)
        reg = reg.contiguous().view(reg.shape[0], -1, 4)

        centerness = centerness.permute(0, 2, 3, 1)
        centerness = centerness.contiguous().view(centerness.shape[0], -1, 1)

        return reg, centerness


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.classifier = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)

        # self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        obj_cls = self.classifier(out)

        # out is B x C x H x W, with C = n_classes + n_anchors
        obj_cls = obj_cls.permute(0, 2, 3, 1)
        batch_size, height, width, channels = obj_cls.shape
        obj_cls = obj_cls.view(batch_size, height, width, self.num_anchors, self.num_classes)
        obj_cls = obj_cls.contiguous().view(x.shape[0], -1, self.num_classes)

        return obj_cls