import os
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from multiprocessing import cpu_count
import time
# from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import numpy as np
from torchsummaryX import summary
import yaml
import cv2

from dataset.dataset import Dataset
from model.atss import ATSSModel, prepare_targets
from model.loss import ATSSLoss
from utils import save_checkpoint, image_tensor_to_array, visualize
from test import detect_single_image, prepare_infer_input, draw_bbox_save_image


def main():
    print('cuda avilable:', torch.cuda.is_available())
    torch.backends.cudnn.benchmark=True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()

    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    train_dataset = Dataset(cfg=cfg, phase='train')
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  num_workers=cpu_count(),
                                  pin_memory=False)
    category_id_to_name = train_dataset.category_id_to_name
    # val_dataset = Dataset(cfg=cfg, phase='val')
    # valloader = data.DataLoader(val_dataset,
    #                             batch_size=24,
    #                             shuffle=False,
    #                             num_workers=cpu_count(),
    #                             pin_memory=False)

    input_size = [3, *cfg['image_size']]
    num_classes = len(cfg['class_names'])
    model = ATSSModel(input_size, num_classes=num_classes)
    status = model.backbone.load_state_dict(torch.load('resnet18-5c106cde.pth'), strict=False)
    print(status)
    model = model.to(device)
    for i in range(len(model.anchors)): model.anchors[i] = model.anchors[i].to(device)

    # summary(model, torch.rand((1, *input_size)))
    criterion = ATSSLoss(num_classes=num_classes)

    optimizer = torch.optim.SGD([{'params': model.parameters()}], momentum=0.9, lr=cfg['lr'], weight_decay=1e-4)
    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr_milestones'], gamma=0.1)

    img_path = 'data/images/sjht/ai-product-injection-mold-inserts/48b02d05ad34/pic/examine/20200904113308-254/camera3_2020-09-04_03_46_47_149830.jpg'
    orig_img, model_input, scale, shift = prepare_infer_input(img_path, cfg['image_size'], train_dataset.to_tensor)


    for i in range(1, cfg['total_epochs']+1):
        epoch_loss = 0

        t = time.time()
        model.train()
        for j, batch in enumerate(trainloader):
            images = batch[0].float().to(device)
            annos = batch[1]

            targets = prepare_targets(annos, model.anchors)

            preds = model(images)

            losses = criterion(preds, targets, model.anchors)
            loss = losses['loss_cls'] + 2.0*losses['loss_reg'] + losses['loss_centerness']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (j+1) % cfg['print_freq'] == 0:
                print('epoch %d, iter %d, train loss:%f, loss_cls:%f, loss_reg:%f, loss_centerness:%f'
                %(i, j+1, loss.item(), losses['loss_cls'].item(), losses['loss_reg'].item(), losses['loss_centerness'].item()))

        print('epoch %d, time %f' %(i, time.time() - t))
        scheduler.step()

        model.eval()
        det_results = detect_single_image(model, model_input.to(device), scale, shift)
        draw_bbox_save_image(orig_img, det_results, 'result_in_train.jpg', category_id_to_name)
        
        if i % cfg['save_freq'] == 0 or i == cfg['total_epochs']:
            save_checkpoint(model, optimizer, cfg['save_model_dir'], i, gpu_num>1)


if __name__ == '__main__':
    main()
