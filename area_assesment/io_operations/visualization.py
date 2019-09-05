import os

import PIL
import cv2
import matplotlib

matplotlib.use('agg')
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from PIL import Image


def just_show_numpy_as_image(numpy_array, type, name):
    if type == 'RGB':
        img = Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
    else:
        img = Image.fromarray(numpy_array.astype(np.uint8))
    img.save(name)
    img.show()


def plot1(img, name=None, show_plot=False, save_output_path=None):
    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap='viridis')
    if show_plot:
        plt.show()
    if save_output_path or save_output_path == '':
        filename = save_output_path + name + '.png'
        logging.info('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=700)


def plot2(img, mask, name=None, overlay=False, alpha=.7, show_plot=False, save_output_path=None, dpi=700):
    if overlay:
        fig = plt.figure(frameon=False, figsize=(10, 10))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        ax.imshow(mask, alpha=alpha, cmap='viridis')
    else:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
        ax[0].imshow(img), ax[0].set_title('1'), ax[0].axis('off')
        ax[1].imshow(mask), ax[1].set_title('2'), ax[1].axis('off')
        fig.tight_layout()
        if name:
            plt.suptitle(name)
    if show_plot:
        plt.show()
    if save_output_path or save_output_path == '':
        # filename = save_output_path + '/' + name + '.png'
        filename = os.path.join(save_output_path, name + '.png')
        logging.info('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=dpi)


def save_mask_and_im_overlayer(img, mask, name=None, save_output_path=None):
    pred_mask = (np.round(mask, 0) * 255).astype(np.uint8)
    val_image = np.round(img * 255, 0).astype(np.uint8)
    pred_mask = np.squeeze(pred_mask)
    pred_mask_red = np.zeros(pred_mask.shape + (3,), np.uint8)
    pred_mask_red[:, :, 0] = pred_mask.copy()
    blended_image = cv2.addWeighted(pred_mask_red, 1, val_image, 1, 0)
    predicted_image = PIL.Image.fromarray(blended_image)
    predicted_image.save(save_output_path)


def plot3(img, mask_true, mask_pred, name=None, show_plot=False, save_output_path=None):
    """
    Plotting raw image, mask true, mask_pred. Support saving into given directory with given filename.

    :param img: numpy array of shape (x, y, ...)
    :param mask_true: numpy array of shape (x, y, ...)
    :param mask_pred: numpy array of shape (x, y, ...)
    :param name:
    :param show_plot:
    :param save_output_path:
    :return:
    """

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    ax[0].imshow(img), ax[0].set_title('1'), ax[0].axis('off')
    ax[1].imshow(mask_true), ax[1].set_title('2'), ax[1].axis('off')
    ax[2].imshow(mask_pred), ax[2].set_title('3'), ax[2].axis('off')
    plt.tight_layout()
    if name:
        plt.suptitle(name)
    if show_plot:
        plt.show()
    if save_output_path:
        filename = save_output_path + name + '.png'
        logging.info('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=500)


def plot_imgs(imgs, name, show_plot=True, save_output_path=''):
    logging.debug('imgs.shape: {}'.format(imgs.shape))
    fig, ax = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1], figsize=(20, 10))
    logging.debug(ax.shape)
    for k, a in enumerate(ax.reshape(-1)):
        i = k // (imgs.shape[0] + 1)
        j = k % imgs.shape[1]
        logging.debug(
            'i:{}/{}, j:{}/{}, imgs[i, j].shape:{}'.format(i, imgs.shape[0], j, imgs.shape[1], imgs[i, j].shape))
        a.imshow(imgs[i, j]), a.set_title('({}, {})'.format(i + 1, j + 1))  # , ax[i, j].axis('off')
    # for i in range(imgs.shape[0]):
    #     for j in range(imgs.shape[1]):
    #         logging.debug('i:{}/{}, j:{}/{}, imgs[i, j].shape:{}'.format(i, imgs.shape[0], j, imgs.shape[1], imgs[i, j].shape))
    #         ax[i, j].imshow(imgs[i, j]) #, ax[i, j].set_title(i+1, j+1), ax[i, j].axis('off')
    plt.tight_layout()

    if save_output_path or save_output_path == '':
        filename = save_output_path + name + '.png'
        logging.info('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=500)

    if show_plot:
        plt.show()
