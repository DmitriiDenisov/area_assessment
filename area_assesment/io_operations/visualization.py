import matplotlib.pyplot as plt


def plot_img(img, name=None, show_plot=True, save_output_path=None):
    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap='viridis')
    if show_plot:
        plt.show()
    if save_output_path:
        filename = save_output_path + name + '.png'
        print('PLOT: {}'.format(filename))
        plt.savefig(filename)


def plot_img_mask(img, mask, name=None, overlay=False, alpha=.7, show_plot=True, save_output_path=None):
    if overlay:
        fig = plt.figure(frameon=False, figsize=(10, 10))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        ax.imshow(mask, alpha=alpha, cmap='viridis')
    else:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
        ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
        ax[1].imshow(mask), ax[1].set_title('mask'), ax[1].axis('off')
        plt.tight_layout()
        if name:
            plt.suptitle(name)
    if show_plot:
        plt.show()
    if save_output_path:
        filename = save_output_path + name + '.png'
        print('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=500)


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
        filename = save_output_path + name + '.png'
        print('PLOT: {}'.format(filename))
        plt.savefig(filename, dpi=500)
