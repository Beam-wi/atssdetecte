import os
import torch
from torch.utils import data
import torchvision
import numpy as np
import yaml
import cv2
import time

from model.atss import prepare_targets, ATSSModel
from dataset.dataset import Dataset
from utils import visualize, image_tensor_to_array


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    dataset = Dataset(cfg=cfg, phase='train')
    category_id_to_name = dataset.category_id_to_name
    category_name_to_id = dataset.category_name_to_id
    data_list = dataset.data_list

    input_size = [3, *cfg['image_size']]
    num_classes = len(cfg['class_names'])
    model = ATSSModel(input_size, num_classes=num_classes)

    # for i, data_anno in enumerate(data_list):
    #     img_path = data_anno['img_path']
    #     img_path = os.path.join(cfg['root_dir'], img_path)
    #     bboxes = data_anno['bboxes']
    #     cls_ids = data_anno['cls_ids']

    #     img = cv2.imread(img_path)
    #     if img is None:
    #         print("Error: read %s fail" % img_path)
    #         exit()        

    #     img = visualize(img, bboxes=bboxes, category_ids=cls_ids, category_id_to_name=category_id_to_name)
    #     cv2.imwrite('img.jpg', img)
    #     time.sleep(1)

    trainloader = data.DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(trainloader):
        # print(batch)
        img, targets = batch

        img = image_tensor_to_array(img[0], cfg['imagenet_default_mean'], cfg['imagenet_default_std'])

        obj_num = targets['obj_num'][0].numpy()
        bboxes = targets['bboxes'][0][:obj_num].numpy()
        category_ids = targets['cls'][0][:obj_num].numpy()
        img = visualize(img, bboxes=bboxes, category_ids=category_ids, category_id_to_name=category_id_to_name)
        cv2.imwrite('img.jpg', img)

        prepared_targets = prepare_targets(targets, model.anchors)
        bbox_cls = prepared_targets[0].squeeze(0)
        
        is_positive = (bbox_cls > -1)
        anchors = torch.cat(model.anchors).numpy()
        anchors = anchors[is_positive]
        bbox_cls = bbox_cls[is_positive].numpy()

        img = visualize(img, bboxes=anchors, category_ids=bbox_cls, category_id_to_name=category_id_to_name)
        cv2.imwrite('anchors.jpg', img)

        time.sleep(1)




