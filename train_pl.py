import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
import yaml
from multiprocessing import cpu_count

from dataset.dataset import Dataset
from model.atss import ATSSModel, prepare_targets
from model.loss import ATSSLoss
from test import detect_single_image, prepare_infer_input, draw_bbox_save_image


class CallBacks(Callback):

    def __init__(self, cfg, image_path, pre_train_weights):
        """
        Args:
            
        """
        super().__init__()
        
        transform = T.Compose([T.ToTensor(),
                               T.Normalize(cfg['imagenet_default_mean'], cfg['imagenet_default_std'])])

        orig_img, model_input, scale, shift = prepare_infer_input(image_path, cfg['image_size'], transform)

        self.orig_img = orig_img
        self.model_input = model_input
        self.scale = scale
        self.shift = shift

    def on_fit_start(self, trainer, pl_module):
        print(pl_module.device)
        for i in range(len(pl_module.model.anchors)): pl_module.model.anchors[i] = pl_module.model.anchors[i].to(pl_module.device)
        status = pl_module.model.backbone.load_state_dict(torch.load('resnet18-5c106cde.pth'), strict=False)
        print(status)

    def on_epoch_end(self, trainer, pl_module):
        # print('on_epoch_end', pl_module.device)
        model = pl_module.model
        model.eval()
        det_results = detect_single_image(model, self.model_input.to(pl_module.device), self.scale, self.shift)
        draw_bbox_save_image(self.orig_img, det_results, 'result_in_train.jpg', pl_module.category_id_to_name)        



class LitATSS(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        input_size = [3, *cfg['image_size']]
        num_classes = len(cfg['class_names'])

        self.cfg = cfg
        self.model = ATSSModel(input_size, num_classes=num_classes)
        self.criterion = ATSSLoss(num_classes=num_classes)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        preds = self.model(x)
        return preds

    # def setup(stage):


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        images = batch[0].float().to(self.device)
        annos = batch[1]
        # print('batch size:', images.shape[0])

        targets = prepare_targets(annos, self.model.anchors, device=self.device)

        preds = self(images)

        losses = self.criterion(preds, targets, self.model.anchors)
        loss = losses['loss_cls'] + 2.0*losses['loss_reg'] + losses['loss_centerness']
            
        loss_dict = {'loss_cls':losses['loss_cls'].item(),
                     'loss_reg':losses['loss_reg'].item(),
                     'loss_centerness':losses['loss_centerness'].item()}

        # if (self.global_step +1) % self.cfg['print_freq'] == 0:
        #     print('epoch %d, iter %d, train loss:%f, loss_cls:%f, loss_reg:%f, loss_centerness:%f'
        #     %(self.current_epoch, self.global_step+1, loss.item(), losses['loss_cls'].item(), losses['loss_reg'].item(), losses['loss_centerness'].item()))

        # Logging to TensorBoard by default
        self.log('train_loss', loss.item())
        self.log_dict(loss_dict, prog_bar=True)
        # self.log('lr', self.trainer.optimizer[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.model.parameters()}], momentum=0.9, lr=self.cfg['lr'], weight_decay=1e-4)
        # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg['lr_milestones'], gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = Dataset(cfg=self.cfg, phase='train')
        self.category_id_to_name = train_dataset.category_id_to_name

        loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=cpu_count(),
        )

        return loader    



def main():

    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    # init model
    model = LitATSS(cfg)

    checkpoint_callback = ModelCheckpoint(dirpath=cfg['checkpoints_dir'],
                                          period=cfg['save_freq'],
                                          save_top_k=-1,
                                          )

    img_path = 'data/images/sjht/ai-product-injection-mold-inserts/48b02d05ad34/pic/examine/20200904113308-254/camera3_2020-09-04_03_46_47_149830.jpg'
    pretrain_weiths = 'resnet18-5c106cde.pth'
    callbacks = CallBacks(cfg, img_path, pretrain_weiths)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=1,
                         benchmark=True,
                         max_epochs=cfg['total_epochs'],
                        #  profiler='simple',
                        #  automatic_optimization=False,
                         callbacks=[checkpoint_callback, callbacks]
                         )
    trainer.fit(model)

if __name__ == '__main__':
    main()