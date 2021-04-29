import os
import sys
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
import numpy as np
import cv2
from PIL import Image
# import matplotlib.pyplot as plt
import math
import albumentations as A
from utils import letterbox_image

MAX_OBJ_NUM = 20
class Dataset(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.phase = phase
        self.size = cfg['image_size']
        self.root_dir = cfg['root_dir']

        cls_names = cfg['class_names']
        self.category_id_to_name = {k: v for k, v in enumerate(cls_names)}
        self.category_name_to_id = {v: k for k, v in self.category_id_to_name.items()}

        if self.phase == 'train':
            self.data_list = self.load_annos(cfg['train_data_file'], self.category_name_to_id)
        else:
            self.data_list = self.load_annos(cfg['val_data_file'], self.category_name_to_id)

        self.aug = A.Compose([
                            #   A.RandomScale(scale_limit=0.1, p=0.5),
                              A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4, rotate_limit=45),
                              A.PadIfNeeded(self.size[0], self.size[1]),
                            #   A.RandomSizedCrop(min_max_height=(int(self.size[0]*0.8), self.size[0]*1.2),
                            #                     height=self.size[0],
                            #                     width=self.size[1],
                            #                     w2h_ratio=self.size[1]/self.size[0]),
                              A.RandomResizedCrop(self.size[0], self.size[1], scale=(0.8, 1.0)),
                              A.IAAPerspective(scale=(0.05, 0.1)),
                              A.Rotate(),
                              A.Flip(),
                              # A.RandomSizedBBoxSafeCrop(height=self.size[1], width=self.size[0]),
                            #   A.RandomBrightnessContrast(p=0.5),
                            #   A.HueSaturationValue(p=0.5),
                              A.ColorJitter()],
                             bbox_params=A.BboxParams(format='pascal_voc',
                                                      label_fields=['cls_ids'],
                                                      min_area=0.3,
                                                      min_visibility=0.3))

        self.to_tensor = T.Compose([T.ToTensor(),
                                    T.Normalize(cfg['imagenet_default_mean'], cfg['imagenet_default_std'])])


    def __getitem__(self, index):
        data_anno = self.data_list[index]
        img_path = data_anno['img_path']
        img_path = os.path.join(self.root_dir, img_path)
        bboxes = data_anno['bboxes']
        cls_ids = data_anno['cls_ids']

        img = cv2.imread(img_path)
        if img is None:
            print("Error: read %s fail" % img_path)
            exit()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # keep ratio resize to input size
        img, scale, shift = letterbox_image(img, self.size)
        bboxes = np.array(bboxes, dtype=np.float)
        bboxes[:, :4] = bboxes[:, :4] * scale
        bboxes[:, [0,2]] = bboxes[:, [0,2]] + shift[0]
        bboxes[:, [1,3]] = bboxes[:, [1,3]] + shift[1]

        # augument
        if self.phase == 'train':
            transformed = self.aug(image=img, bboxes=bboxes, cls_ids=cls_ids)
            img = Image.fromarray(transformed['image'])
            bboxes = np.array(transformed['bboxes'])
            cls_ids = np.array(transformed['cls_ids'])
            
        obj_num = len(bboxes)

        if obj_num == 0:
            print('obj_num == 0')
            bboxes = np.zeros((MAX_OBJ_NUM, 4), dtype=float)
            cls_ids = np.zeros(MAX_OBJ_NUM, dtype=int)
        elif obj_num < MAX_OBJ_NUM:            
            bboxes = np.pad(bboxes, ((0, MAX_OBJ_NUM - obj_num), (0,0)))
            cls_ids = np.pad(cls_ids, (0, MAX_OBJ_NUM - obj_num))
            

        scale_shift = torch.Tensor([scale] + shift)
        # to tensor and normalize
        img = self.to_tensor(img)
        targets = {'bboxes':bboxes, 'cls':cls_ids, 'obj_num':obj_num, 'transform':scale_shift, 'img_path':img_path}
        
        return img, targets

    def __len__(self):
        return len(self.data_list)

    def load_annos(self, anno_file, class_name2idx):
        anno_list = []
        with open(anno_file, 'r') as fd:
            lines = fd.readlines()

        if self.phase == 'train':
            start_idx = 1
        else:
            start_idx = 0

        for line in lines:
            anno = {'img_path':'', 'bboxes':[], 'cls_ids':[], 'cls_labels':[]}
            splits = line.split()
            anno['img_path'] = splits[start_idx]
            for i in range(start_idx+1,len(splits), 5):
                cls_name = splits[i].split('_')[0]
                anno['cls_labels'].append(cls_name)
                cls_idx = class_name2idx[cls_name]
                anno['cls_ids'].append(cls_idx)
                bbox = splits[i+1: i+5]
                anno['bboxes'].append(bbox)
            anno_list.append(anno)
        return anno_list
