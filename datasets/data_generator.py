import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .keypoint_encoder import KeypointEncoder


class FashionAIKeypoints(Dataset):

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        self.encoder = KeypointEncoder()

        # Read csv
        # TODO: Need a cleaner way to combine pathlib and path in pd columns.
        if phase == 'test':
            data_dir = opt.db_path / 'r2_test_b/'
            anno_df = pd.read_csv(data_dir / 'test.csv')
            anno_df['image_path'] = str(data_dir) + '/' + anno_df['image_id']
        else:
            data_dir0 = opt.db_path / 'wu_train/'
            anno_df0 = pd.read_csv(data_dir0 / 'Annotations/annotations.csv')
            anno_df0['image_path'] = str(data_dir0) + '/' + anno_df0['image_id']

            data_dir1 = opt.db_path / 'r1_train/'
            anno_df1 = pd.read_csv(data_dir1 / 'Annotations/train.csv')
            anno_df1['image_path'] = str(data_dir1) + '/' + anno_df1['image_id']

            data_dir2 = opt.db_path / 'r1_test_a/'
            anno_df2 = pd.read_csv(data_dir2 / 'fashionAI_key_points_test_a_answer_20180426.csv')
            anno_df2['image_path'] = str(data_dir2) + '/' + anno_df2['image_id']

            data_dir3 = opt.db_path / 'r1_test_b/'
            anno_df3 = pd.read_csv(data_dir3 / 'fashionAI_key_points_test_b_answer_20180426.csv')
            anno_df3['image_path'] = str(data_dir3) + '/' + anno_df3['image_id']
            anno_df3_train, anno_df3_val = train_test_split(anno_df3, test_size=0.2, random_state=42)

            if phase == 'train':
                anno_df = pd.concat([anno_df0, anno_df1, anno_df2, anno_df3_train])
                # anno_df = anno_df3_train.iloc[:100] # for debug
            else:
                anno_df = anno_df3_val

        # Keep only the direct image_path and relevant keypoints
        # I should drop all the unncessary category over here to save memory.
        self.anno_df = anno_df[anno_df['image_category'] == self.opt.category][['image_path'] + self.opt.keypoints]

    def __getitem__(self, idx):
        """
        Returns one images per __getitem__
        """
        img = cv2.imread(self.get_image_path(idx))  # BGR
        img_h, img_w, _ = img.shape
        kpts = self.get_keypoints(idx)

        # It prints a matrix like this for 'skirt'
        # >> kpts
        # [[ 83 112   1]
        #  [326 120   1]
        #  [ 57 405   1]
        #  [342 398   1]]

        # Image Transfomations
        if self.phase == 'train':

            # IT #1: Horizontal flip
            random_flip = np.random.randint(0, 2)

            # Apply horizontal flip if random_flip=1
            if random_flip == 1:
                img = cv2.flip(img, 1)

                # Calculate the flipped horizontal keypoints
                kpts[:, 0] = img_w - kpts[:, 0]

                # Since the image is flipped, we'll need to flip the
                # keypoints position in the vector too.
                for conj in self.opt.conjug:
                    kpts[conj] = kpts[conj[::-1]]

            # IT #2: Rotation
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D(center=(img_w / 2, img_h / 2), angle=angle, scale=1)
            # `dsize` is the size of the output image.
            img = cv2.warpAffine(img, M, dsize=(img_w, img_h), flags=cv2.INTER_CUBIC)
            kpts_tmp = kpts.copy()
            kpts_tmp[:, 2] = 1
            kpts[:, :2] = np.matmul(kpts_tmp, M.T)

        # IT #3: Scale images to img_max_size
        scale = self.opt.img_max_size / max(img_w, img_h)
        img_h2, img_w2 = int(img_h * scale), int(img_w * scale)
        img = cv2.resize(img, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
        kpts[:, :2] = kpts[:, :2] * scale
        kpts = kpts.astype(np.float32)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
        img[[0, 2]] = img[[2, 0]]  # BGR to RGB

        # IT #4: Image normalization
        # TODO: did it normalize correctly?
        img = img / 255.0
        img = (img - self.opt.mu) / self.opt.sigma

        return torch.from_numpy(img), torch.from_numpy(kpts)

    def collate_fn(self, batch):
        ''' `collate_fn` allows us to specify how exactly the samples
        need to be batched in `torch.utils.data.DataLoader`. Note that default
        collate should work fine for most use cases.
        '''
        imgs, kpts = zip(*batch)

        # We want the images to have the same size to accelerate the dynamic
        # graph.
        pad_imgs = torch.zeros(len(imgs), 3, self.opt.img_max_size, self.opt.img_max_size)
        heatmaps, vis_masks = [], []

        for i, img in enumerate(imgs):
            # If the image is smaller than `img_max_size`, we pad it with 0s.
            # This allows all images to have the same size.
            pad_imgs[i, :, :img.size(1), :img.size(2)] = img

            # For each image, create heatmaps and visibility masks.
            img_heatmaps, img_vis_masks = self.encoder.encode(kpts[i],
                                                              self.opt.img_max_size,
                                                              self.opt.hm_stride,
                                                              self.opt.hm_alpha,
                                                              self.opt.hm_sigma)

            # TODO: Can I avoid appending and do everything in torch?
            heatmaps.append(img_heatmaps)
            vis_masks.append(img_vis_masks)

        heatmaps = torch.stack(heatmaps)    # [batch_size, num_keypoints, h, w]
        vis_masks = torch.stack(vis_masks)  # [batch_size, num_keypoints]

        if self.phase == 'test':
            return pad_imgs, heatmaps, vis_masks, kpts

        return pad_imgs, heatmaps, vis_masks

    def __len__(self):
        return len(self.anno_df)

    def get_image_path(self, image_index):
        row = self.anno_df.iloc[image_index]
        return row['image_path']

    def get_keypoints(self, image_index):
        row = self.anno_df.iloc[image_index][self.opt.keypoints]
        return np.array([np.fromstring(x, dtype=int, sep='_') for x in row])
