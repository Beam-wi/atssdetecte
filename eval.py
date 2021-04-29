import torch
from torch.utils import data
import cv2
import yaml
import numpy as np
import time

from model.atss import ATSSModel
from dataset.dataset import Dataset
from test import detect_single_image, detect_images
from mean_ap import eval_map
from train_pl import LitATSS


def main():
    print('cuda avilable:', torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()

    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    dataset = Dataset(cfg=cfg, phase='val')
    valloader = data.DataLoader(dataset, batch_size=16)

    category_id_to_name = dataset.category_id_to_name
    category_name_to_id = dataset.category_name_to_id
    data_list = dataset.data_list

    # input_size = [3, *cfg['image_size']]
    # num_classes = len(cfg['class_names'])
    # model = ATSSModel(input_size, num_classes=num_classes)
    # model_path = 'checkpoints/1027/160/model.pth'
    # status = model.load_state_dict(torch.load(model_path, map_location=device))
    # print(status)

    checkpoint_path = 'checkpoints/1027/epoch=159.ckpt'
    lit_model = LitATSS(cfg).load_from_checkpoint(checkpoint_path=checkpoint_path)
    model = lit_model.model

    model.to(device)
    model.eval()

    det_results = []
    annotations = []
    t2 = time.time()
    for i, batch in enumerate(valloader):
        input_images = batch[0].to(device)
        annos = batch[1]
        scale_shift = annos['transform']

        t1 = time.time()
        # det_result = detect_single_image(model, input_images, scale_shift[0][0], scale_shift[0][1:3])
        det_result = detect_images(model, input_images, scale_shift[:, 0], scale_shift[:, 1:3])
        det_results.extend(det_result)
        # print(len(det_result))
        print('detect_images time:', time.time() - t1)

        for j in range(input_images.shape[0]):

            gt_num = annos['obj_num'][j]

            gt_bboxes_per_image = annos['bboxes'][j][:gt_num]
            gt_bboxes_per_image[:, [0, 2]] = gt_bboxes_per_image[:, [0, 2]] - scale_shift[j][1]
            gt_bboxes_per_image[:, [1, 3]] = gt_bboxes_per_image[:, [1, 3]] - scale_shift[j][2]
            gt_bboxes_per_image = gt_bboxes_per_image / scale_shift[j][0]
            gt_bboxes_per_image = gt_bboxes_per_image.numpy()

            gt_labels = annos['cls'][j][:gt_num].numpy()

            annotations.append({'bboxes': gt_bboxes_per_image, 'labels': gt_labels})
    print('total detect time:', time.time() - t2)

    mean_ap, eval_results = eval_map(det_results, annotations, iou_thr=0.75)        



if __name__ == '__main__':
    main()