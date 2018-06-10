import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch


def draw_keypoint_with_caption(image, keypoint, text):
    '''
    :param image:
    :param keypoint: [x, y]
    :param text: string
    :return: image
    '''
    alpha = 0.5
    color1 = (0, 255, 0)
    thick = 2
    l = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    overlay = image.copy()
    x, y = keypoint
    overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color1, thick)
    overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color1, thick)
    overlay = cv2.putText(overlay, text, (0, image.shape[0]), font, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def draw_keypoints(image, keypoints, gt_keypoints=None):
    '''
    :param image:
    :param keypoints: [[x, y, v], ...]
    :return:
    '''
    alpha = 0.5
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
    thick = 1
    l = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    overlay = image.copy()
    if gt_keypoints is None:
        for kpt in keypoints:
            x, y, v = kpt
            if v > 0:
                overlay = cv2.line(overlay, (x-l, y-l), (x+l, y+l), color1, thick)
                overlay = cv2.line(overlay, (x-l, y+l), (x+l, y-l), color1, thick)

    if gt_keypoints is not None:
        for k in range(len(keypoints)):
            gtx, gty, gtv = gt_keypoints[k]
            x, y, v = keypoints[k]
            if gtv > 0:
                overlay = cv2.line(overlay, (x - l, y - l), (x + l, y + l), color1, thick)
                overlay = cv2.line(overlay, (x - l, y + l), (x + l, y - l), color1, thick)
                overlay = cv2.putText(overlay, str(k), (x, y), font, font_scale, color1, thick, cv2.LINE_AA)
                overlay = cv2.line(overlay, (gtx - l, gty - l), (gtx + l, gty + l), color2, thick)
                overlay = cv2.line(overlay, (gtx - l, gty + l), (gtx + l, gty - l), color2, thick)
                overlay = cv2.putText(overlay, str(k), (gtx, gty), font, font_scale, color2, thick, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def draw_heatmap(image, heatmap):
    '''
    :param image:
    :param heatmap:
    :param save_path:
    :return:
    '''
    hp_max = np.amax(heatmap)
    scale = 1
    if hp_max != 0:
        scale = 255 // hp_max
    heatmap = (heatmap * scale).astype(np.uint8)
    alpha = 0.7
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, alpha, image, 1 - alpha, 0)
    return fin

def normalized_error(preds, targets, widths):
    '''
    :param preds: [[x, y, v], ...]
    :param targets: [[x, y, v], ...]
    :param widths: [[w1], [w2], ...]
    :return:
    '''
    dist = preds[:, :2] - targets[:, :2]
    dist = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2)
    targets = np.copy(targets)
    targets[targets<0] = 0
    if np.sum(targets[:, 2]) == 0:
        return 0
    ne = np.sum(dist/widths * targets[:, 2]) / np.sum(targets[:, 2])
    return ne

def get_free_id():
    import pynvml

    pynvml.nvmlInit()
    def get_free_ratio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        return ratio

    device_count = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(device_count):
        if get_free_ratio(i) < 70:
            available.append(i)

    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus

def set_gpu(gpu_input):
    free_ids = get_free_id()

    if gpu_input == 'all':
        gpus = free_ids
    else:
        gpus = gpu_input
        if any([g not in free_ids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')

    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def data_frame_template():
    df = pd.DataFrame(columns=['image_id','image_category','neckline_left','neckline_right','center_front','shoulder_left',
                               'shoulder_right','armpit_left','armpit_right','waistline_left','waistline_right',
                               'cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left',
                               'top_hem_right','waistband_left','waistband_right','hemline_left','hemline_right',
                               'crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out'])
    return df

def compute_keypoints(opt, img0, net, encoder, doflip=False):
    img_h, img_w, _ = img0.shape

    # Scale images to img_max_size
    scale = opt.img_max_size / max(img_w, img_h)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)
    img = cv2.resize(img0, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
    img[[0, 2]] = img[[2, 0]]
    img = img / 255.0
    img = (img - opt.mu) / opt.sigma

    pad_imgs = np.zeros([1, 3, opt.img_max_size, opt.img_max_size], dtype=np.float32)
    pad_imgs[0, :, :img_h2, :img_w2] = img
    data = torch.from_numpy(pad_imgs)
    data = data.cuda(async=True)

    _, hm_pred = net(data)
    hm_pred = F.relu(hm_pred, False)
    hm_pred = hm_pred[0].data.cpu().numpy()

    if doflip:
        a = np.zeros_like(hm_pred)
        a[:, :, :img_w2 // opt.hm_stride] = np.flip(hm_pred[:, :, :img_w2 // opt.hm_stride], 2)
        for conj in opt.conjug:
            a[conj] = a[conj[::-1]]
        hm_pred = a

    return hm_pred
