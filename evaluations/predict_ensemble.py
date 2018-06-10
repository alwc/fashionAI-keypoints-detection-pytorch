import argparse
import os
import math
import sys
sys.path.append("..")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from tqdm import tqdm

import models
import utils
from datasets import FashionAIKeypoints
from loss import CPNLoss
from utils import LRScheduler
from utils.config import opt


def main(**kwargs):
    opt._parse(kwargs)
    # n_gpu = utils.set_gpu(args.gpu)

    val_dataset = FashionAIKeypoints(opt, phase='val')
    encoder = val_dataset.encoder
    nes = []

    print('Evaluating: {}'.format(opt.category))
    print('Validation sample number: {}'.format(len(val_dataset)))
    cudnn.benchmark = True

    net1 = getattr(models, opt.model[0])(opt)
    checkpoint = torch.load(opt.load_checkpoint_path)    # Must be before cuda
    net1.load_state_dict(checkpoint['state_dict'])
    net1 = net1.cuda()
    # net1 = DataParallel(net)
    net1.eval()

    net2 = getattr(models, opt.model[1])(opt)
    checkpoint = torch.load(opt.load_checkpoint_path_2)  # Must be before cuda
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.cuda()
    # net2 = DataParallel(net2)
    net2.eval()

    for idx in tqdm(range(len(val_dataset))):
        img_path = val_dataset.get_image_path(idx)
        kpts = val_dataset.get_keypoints(idx)
        img0 = cv2.imread(img_path)  # BGR
        img0_flip = cv2.flip(img0, 1)
        img_h, img_w, _ = img0.shape

        scale = opt.img_max_size / max(img_w, img_h)

        hm_pred = utils.compute_keypoints(opt, img0, net1, encoder)
        hm_pred2 = utils.compute_keypoints(opt, img0_flip, net1, encoder, doflip=True)
        hm_pred3 = utils.compute_keypoints(opt, img0, net2, encoder)
        hm_pred4 = utils.compute_keypoints(opt, img0_flip, net2, encoder, doflip=True)

        x, y = encoder.decode_np(hm_pred + hm_pred2 + hm_pred3 + hm_pred4,
                                 scale, opt.hm_stride, (img_w/2, img_h/2))
        keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)

        if args.visual:
            kpt_img = utils.draw_keypoints(img0, keypoints)
            save_img_path = str(opt.db_path / 'tmp/ensemble_{0}{1}.png'.format(opt.category, idx))
            cv2.imwrite(save_img_path, kpt_img)

        left, right = opt.datum
        x1, y1, v1 = kpts[left]
        x2, y2, v2 = kpts[right]

        if v1 == -1 or v2 == -1:
            continue

        width = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        ne = utils.normalized_error(keypoints, kpts, width)
        nes.append([ne])

    nes = np.array(nes)
    print(np.mean(nes, axis=0))

if __name__ == '__main__':
    import fire
    fire.Fire()
