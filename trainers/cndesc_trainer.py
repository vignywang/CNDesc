#
# Created by ZhangYuyang on 2019/9/18
#
# 训练算子基类
import os
import time
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
import numpy as np
from nets import get_model
from data_utils import get_dataset
from trainers.base_trainer import BaseTrainer
from utils.utils import spatial_nms
from utils.utils import DescriptorTripletLoss_E,Consistent_loss
from utils.utils import PointHeatmapWeightedBCELoss
from torch.autograd import Variable as V
from thop import profile

class CNDescTrainer(BaseTrainer):

    def __init__(self, **config):
        super(CNDescTrainer, self).__init__(**config)
        self.height=self.config['train']['height']
    def _initialize_dataset(self):
        # 初始化数据集
        self.logger.info('Initialize {}'.format(self.config['train']['dataset']))
        self.train_dataset = get_dataset(self.config['train']['dataset'])(**self.config['train'])

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            drop_last=True
        )
        self.epoch_length = len(self.train_dataset) // self.config['train']['batch_size']

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])()
       # self.logger.info("Initialize network arch {}".format(self.config['model']['extractor']))
       # extractor = get_model(self.config['model']['extractor'])()
        input = torch.randn(1, 3, 640, 480).cuda()
        input = V(input).to(self.device)
        flops, params = profile(model.cuda(), inputs=(input,))
        # print(flops)
        # print(params)
        print("Number of flops: %.2fGFLOPs" % (flops / 1e9))
        print("Number of parameter: %.2fM" % (params / 1e6))
        exit(0)

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        #self.extractor = extractor.to(self.device)

        #if self.config['train']['pretrain_path'] != '':
         #   self._load_model_params(self.config['train']['pretrain_path'],self.model)

    def _initialize_loss(self):
        # 初始化描述子loss
        self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
        self.descriptor_loss = DescriptorTripletLoss_E(self.device,margin=1)
        self.consistent_loss = Consistent_loss()

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):

        # 初始化学习率调整算子
        if self.config['train']['lr_mod']=='LambdaLR':
            self.logger.info("Initialize lr_scheduler of LambdaLR: (%d, %d)" % (self.config['train']['maintain_epoch'], self.config['train']['decay_epoch']))
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch  - self.config['train']['maintain_epoch']) / float(self.config['train']['decay_epoch'] + 1)
                return lr_l
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        else:
            milestones = [20, 30]
            self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d)" % (milestones[0], milestones[1]))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        print("Load pretrained model %s " % ckpt_file)

        model_dict = previous_model.state_dict()
        pretrain_dict = torch.load(ckpt_file, map_location=self.device)
        model_dict.update(pretrain_dict)
        previous_model.load_state_dict(model_dict)
        return previous_model

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")


    def _train_func(self, epoch_idx):
        self.model.train()
        stime = time.time()
        total_loss = 0
        for i, data in enumerate(self.train_dataloader):

            # 读取相关数据
            image = data["image"].to(self.device)
            desp_point = data["desp_point"].to(self.device)
            warped_image = data["warped_image"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)
            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)
            image_pair = torch.cat((image, warped_image), dim=0)
            # 模型预测
            descriptor = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            descriptor = f.grid_sample(descriptor, desp_point_pair, mode="bilinear", padding_mode="border")[:, :, :,0].transpose(1, 2)
            desp0, desp1 = torch.chunk(descriptor, 2, dim=0)

            desp_loss = self.descriptor_loss(desp0, desp1, valid_mask, not_search_mask)
            consistent_loss = self.consistent_loss(desp0, desp1, valid_mask,
                                                   not_search_mask)  # self.consistent_loss(desp0, desp1, valid_mask, not_search_mask)

            loss = desp_loss + consistent_loss* self.config['train']['loss_weight']

            total_loss += loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if i % self.config['train']['log_freq'] == 0:

                loss0 = desp_loss.item()
                loss1 = consistent_loss.item()
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, desp_loss1 = %.4f, desp_loss2 = %.4f."
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        loss0,
                        loss1,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()
        self.logger.info("Total_loss:" + str(total_loss.detach().cpu().numpy()))
        # save the model
        if self.multi_gpus:
            torch.save(
                self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
         #   torch.save(
            #    self.extractor.module.state_dict(),os.path.join(self.config['ckpt_path'], 'extractor_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
          #  torch.save(self.extractor.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_%02d.pt' % epoch_idx))











