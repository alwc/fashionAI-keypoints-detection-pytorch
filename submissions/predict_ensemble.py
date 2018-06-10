import argparse
import os
import math
import sys
sys.path.append("..")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    # n_gpu = utils.set_gpu(opt.gpu)

    test_dataset = FashionAIKeypoints(opt, phase='test')
    encoder = test_dataset.encoder
    df = utils.data_frame_template()

    print('Testing: {}'.format(opt.category))
    print('Testing sample number: {}'.format(len(val_dataset)))
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

    for idx in tqdm(range(len(test_dataset))):
        img_path = test_dataset.get_image_path(idx)
        img0 = cv2.imread(img_path)  # BGR
        img0_flip = cv2.flip(img0, 1)
        img_h, img_w, _ = img0.shape

        scale = opt.img_max_size / max(img_w, img_h)

        hm_pred = utils.compute_keypoints(opt, img0, net, encoder)
        hm_pred_flip = utils.compute_keypoints(opt, img0_flip, net, encoder, doflip=True)
        hm_pred2 = utils.compute_keypoints(opt, img0, net2, encoder)
        hm_pred_flip2 = utils.compute_keypoints(opt, img0_flip, net2, encoder, doflip=True)

        x, y = encoder.decode_np(hm_pred + hm_pred_flip + hm_pred2 + hm_pred_flip2,
                                 scale, opt.hm_stride, (img_w/2, img_h/2))
        keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)


        row = test_dataset.anno_df.iloc[idx]
        df.at[idx, 'image_id'] = row['image_id']
        df.at[idx, 'image_category'] = row['image_category']

        for k, kpt_name in enumerate(opt.keypoints[opt.category]):
            df.at[idx, kpt_name] = str(keypoints[k,0])+'_'+str(keypoints[k,1])+'_1'

        if args.visual:
            kp_img = utils.draw_keypoints(img0, keypoints)
            save_img_path = str(opt.db_path / 'tmp/ensemble_{0}{1}.png'.format(opt.category, idx))
            cv2.imwrite(save_img_path, kpt_img)

    df.fillna('-1_-1_-1', inplace=True)
    print(df.head(5))
    df.to_csv(opt.pred_path / 'ensemble_{}.csv'.format(opt.category), index=False)

if __name__ == '__main__':
    import fire
    fire.Fire()
