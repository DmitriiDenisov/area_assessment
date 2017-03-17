import matplotlib.pyplot as plt


def plot_img_mask(img, mask):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask), ax[1].set_title('mask'), ax[1].axis('off')
    plt.tight_layout(), plt.show()


def plot_img_mask_pred(img, mask_true, mask_pred, name=None, show_plot=True, save_output_path=None):
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
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask_true), ax[1].set_title('mask_true'), ax[1].axis('off')
    ax[2].imshow(mask_pred), ax[2].set_title('mask_pred'), ax[2].axis('off')
    plt.tight_layout()
    if name:
        plt.suptitle(name)
    if show_plot:
        plt.show()
    if save_output_path:
        print('PLOT: {}'.format(save_output_path + name + '.pdf'))
        fig.savefig(save_output_path + name + '.pdf', dpi=500)
