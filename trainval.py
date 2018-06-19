import argparse
import os
import sys
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import utils
from datasets import FashionAIKeypoints
from loss import CPNLoss
from utils import LRScheduler
from utils.config import opt


def print_log(category, epoch, lr, train_metrics, train_time, val_metrics=None, val_time=None, save_dir=None, log_mode=None):
    if epoch > 1:
        log_mode = 'a'
    train_metrics = np.mean(train_metrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str1 = 'Train:      time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
           % (train_time, train_metrics[0], train_metrics[1], train_metrics[2])
    print(str0)
    print(str1)
    f = open(save_dir + 'kpt_' + category + '_train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str1 + '\n')
    if val_time is not None:
        val_metrics = np.mean(val_metrics, axis=0)
        str2 = 'Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
               % (val_time, val_metrics[0], val_metrics[1], val_metrics[2])
        print(str2 + '\n')
        f.write(str2 + '\n\n')
    f.close()


def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (input_imgs, heatmaps, vis_masks) in enumerate(data_loader):
        print("train loop ", i)
        input_imgs = input_imgs.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        vis_masks = vis_masks.cuda(async=True)

        # `hm_global_pred` is the P2 output from GlobalNet and `hm_refine_pred`
        # is the concantenated output across all layer from RefineNet.
        #
        # Both tensors have size [batch_size, num_keypoints, 128, 128].
        hm_global_preds, hm_refine_preds = net(input_imgs)
        total_loss, global_loss, refine_loss = loss(heatmaps,
                                                    hm_global_preds,
                                                    hm_refine_preds,
                                                    vis_masks)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        metrics.append([total_loss.item(), global_loss.item(), refine_loss.item()])

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time

def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []

    for i, (input_imgs, heatmaps, vis_masks) in enumerate(data_loader):
        input_imgs = input_imgs.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        vis_masks = vis_masks.cuda(async=True)

        hm_global_preds, hm_refine_preds = net(input_imgs)
        total_loss, global_loss, refine_loss = loss(heatmaps,
                                                    hm_global_preds,
                                                    hm_refine_preds,
                                                    vis_masks)

        metrics.append([total_loss.item(), global_loss.item(), refine_loss.item()])

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time

def main(**kwargs):
    # 1. Parse command line arguments.
    opt._parse(kwargs)

    # 2. Visdom
    # vis = Visualizer(env=opt.env)

    # 3. GPU settings
    # n_gpu = utils.set_gpu('0,1')

    # 4. Configure model
    print('==> Traing model for clothing type: {}'.format(opt.category))
    cudnn.benchmark = True
    net = getattr(models, opt.model)(opt)

    # 5. Initialize checkpoints directory
    lr = opt.lr
    save_dir = opt.checkpoint_path
    resume = False

    start_epoch = 1
    best_val_loss = float('inf')
    log_mode = 'w'

    if opt.load_checkpoint_path:
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(opt.load_checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    # 6. Data setup
    train_dataset = FashionAIKeypoints(opt, phase='train')
    print('Train sample number: {}'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)

    val_dataset = FashionAIKeypoints(opt, phase='val')
    print('Val sample number: {}'.format(len(val_dataset)))
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            collate_fn=val_dataset.collate_fn,
                            pin_memory=True)

    net = net.cuda()
    # net = DataParallel(net)
    loss = CPNLoss()
    loss = loss.cuda()

    # 7. Loss, optimizer and LR scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    lrs = LRScheduler(lr, patience=3, factor=0.1, min_lr=0.01*lr, best_loss=best_val_loss)

    # 8. Training loop
    for epoch in range(start_epoch, opt.max_epochs + 1):
        # Training
        print("Start training loop...")
        train_metrics, train_time = train(train_loader, net, loss, optimizer, lr)

        # Validating
        print("Start validating loop...")
        with torch.no_grad():
            val_metrics, val_time = validate(val_loader, net, loss)

        print_log(opt.category, epoch, lr, train_metrics, train_time,
                  val_metrics, val_time, save_dir=save_dir, log_mode=log_mode)

        val_loss = np.mean(val_metrics[:, 0])
        lr = lrs.update_by_rule(val_loss)

        # Save checkpoints
        if val_loss < best_val_loss or epoch % 10 == 0 or lr is None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.module.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss},
                opt.checkpoint_path / 'kpt_{}_{:03d}.ckpt'.format(opt.category, epoch))

        if lr is None:
            print('Training is early-stopped')
            break

if __name__ == '__main__':
    import fire
    fire.Fire()
