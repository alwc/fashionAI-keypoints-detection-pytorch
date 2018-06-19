import logging
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
from utils import LRScheduler, initialize_logger, log_model
from utils.config import opt


def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (input_imgs, heatmaps, vis_masks) in enumerate(data_loader):
        logging.info("train loop ", i)
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
    logging.info('==> Traing model for clothing type: {}'.format(opt.category))
    cudnn.benchmark = True
    net = getattr(models, opt.model)(opt)

    # 5. Initialize logger
    cur_time = time.strftime('%Y-%m-%dT%H:%M:%S', timm.localtime())
    initialize_logger(f'{opt.category}_{opt.model}_{cur_time}')

    # 6. Initialize checkpoints directory
    lr = opt.lr
    start_epoch = 1
    best_val_loss = float('inf')

    if opt.load_checkpoint_path:
        logging.info('==> Resuming from checkpoint...')
        checkpoint = torch.load(opt.load_checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])

    # 7. Data setup
    train_dataset = FashionAIKeypoints(opt, phase='train')
    logging.info('Train sample number: {}'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)

    val_dataset = FashionAIKeypoints(opt, phase='val')
    logging.info('Val sample number: {}'.format(len(val_dataset)))
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

    # 8. Loss, optimizer and LR scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    lrs = LRScheduler(lr, patience=3, factor=0.1, min_lr=0.01*lr, best_loss=best_val_loss)

    # 9. Training loop
    for epoch in range(start_epoch, opt.max_epochs + 1):
        # Training
        logging.info("Start training loop...")
        train_metrics, train_time = train(train_loader, net, loss, optimizer, lr)

        # Validating
        logging.info("Start validating loop...")
        with torch.no_grad():
            val_metrics, val_time = validate(val_loader, net, loss)

        log_model(epoch, lr, train_metrics, train_time, val_metrics, val_time)

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
                'save_dir': opt.checkpoint_path,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss
            }, opt.checkpoint_path / 'kpt_{}_{:03d}.ckpt'.format(opt.category, epoch))

        if lr is None:
            logging.info('Training is early-stopped')
            break

if __name__ == '__main__':
    import fire
    fire.Fire()
