from keras import backend as K


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    smooth = 1e-12
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)
