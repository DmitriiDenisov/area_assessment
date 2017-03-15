import matplotlib.pyplot as plt


def plot_img_mask(img, mask):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask), ax[1].set_title('mask_true'), ax[1].axis('off')
    plt.tight_layout(), plt.show()


def plot_img_mask_pred(img, mask_true, mask_pred):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 10))
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask_true), ax[1].set_title('mask_true'), ax[1].axis('off')
    ax[2].imshow(mask_pred), ax[2].set_title('mask_pred'), ax[2].axis('off')
    plt.tight_layout(), plt.show()
