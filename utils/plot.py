import os
import sys
sys.path.append("..")

import cv2
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects

from utils.config import opt

# Keypoints by clothing categories
num_kp_categories = sorted(list(set().union(*opt.kp_dict.values())))
kp2idx = {kp: i for i, kp in enumerate(num_kp_categories)}
idx2kp = {v: k for k, v in kp2idx.items()}


# Annotation code
def draw_outline(plt_obj, linewidth):
    plt_obj.set_path_effects([
        patheffects.Stroke(linewidth=linewidth, foreground='black'),
        patheffects.Normal()
    ])


def draw_circles(ax, xy_pair, color):
    patch = ax.add_patch(
        patches.Circle(xy_pair, radius=10, fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)


def annotate_kp(ax, x, y, kp, color, show_text=False):
    kp_txt = '{}: {}'.format(kp2idx[kp], kp) if show_text else kp2idx[kp]
    rotation = -20 if show_text else 0

    text = ax.text(
        x,
        y,
        kp_txt,
        rotation=rotation,
        verticalalignment='top',
        color=color,
        fontsize=10,
        weight='bold')

    draw_outline(text, 2)


def annotate_dne_kps(ax, kps, show_text=False):
    kp_idx = [kp2idx[kp] for kp in kps]

    if show_text:
        kp_tuple = list(zip(kp_idx, kps))
        txt = 'DNE:\n' + '\n'.join(
            ['({}) {}'.format(idx, kp) for idx, kp in kp_tuple])
    else:
        txt = 'DNE: ' + ', '.join(map(str, kp_idx))

    text = ax.text(
        10,
        10,
        txt,
        rotation=0,
        verticalalignment='top',
        color='yellow',
        fontsize=10,
        weight='bold')

    draw_outline(text, 2)


# Plotting code
def plot_kps(ax, row, show_text):
    """
    Plot the key points on the image
    """
    # Record which key points do not exist
    dne_kps = []

    for kp in opt.kp_dict[row['image_category']]:
        x, y, visibility_type = map(int, row[kp].split('_'))

        # 1  = Visible (white)
        # 0  = Not visible (red)
        # -1 = Does not exist: it means this kp belongs to this category
        # but the annotation does not exist for this image
        if visibility_type == 1:
            draw_circles(ax, (x, y), 'white')
            annotate_kp(ax, x, y, kp, 'white', show_text)
        elif visibility_type == 0:
            draw_circles(ax, (x, y), 'red')
            annotate_kp(ax, x, y, kp, 'red', show_text)
        else:
            dne_kps.append(kp)

    if dne_kps:
        annotate_dne_kps(ax, dne_kps, show_text)


def plot_img(dir_path, ax, row, show_text):
    """
    Plot the image
    """
    img_category = row['image_category']
    img_id = row['image_id'].split('/')[-1][:-4]
    ax.set_title("{}\n{}".format(img_category, img_id))

    im = cv2.imread(str(dir_path / row['image_id']))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax.imshow(im)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    plot_kps(ax, row, show_text)


def plot_img_with_kps(dir_path, df, num_per_row=3, figsize=(16, 16), show_text=False):
    fig, axes = plt.subplots(
        nrows=num_per_row,
        ncols=-(-df.shape[0] // num_per_row),
        figsize=figsize)

    print(df['image_id'].tolist())

    if df.shape[0] > 1:
        axes = axes.flatten()

        for i, (_, row) in enumerate(df.iterrows()):
            ax = axes[i]
            plot_img(dir_path, ax, row, show_text)
    else:
        plot_img(dir_path, axes, df.iloc[0], show_text)


def plot_img_by_id(dir_path, df, img_id):
    row_df = df[df['image_id'].str.contains(img_id)]
    img_category = row_df['image_category'].iloc[0]
    display(row_df[opt.kp_dict[img_category]])
    plot_img_with_kps(dir_path, row_df, num_per_row=1, figsize=(9, 9), show_text=True)
