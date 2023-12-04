import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError


def imagesc(img, pos, scale_fig, title, version = 0, cmap = "gray"):

    img = img[pos,:,:]
    plt.figure(figsize=(4, 3))
    plt.imshow(img, cmap=cmap, vmin=scale_fig[0], vmax=scale_fig[1])
    plt.title(title)
    plt.axis('off')
    plt.show()

def normalize_image(img):

    img = img / tfp.stats.percentile(img, 99)
    return img

def synthesize_T2_Weighted(T1_map, T2_map, PD_map,s):
    TE = 85
    TR = 4000

    tf.cast(TE, dtype=tf.float64)
    tf.cast(TR, dtype=tf.float64)

    im_tmp = PD_map * (1-tf.math.exp(-TR /T1_map)) * tf.math.exp(-TE /T2_map)
    im_syn_T2w = s*im_tmp

    return im_syn_T2w

def synthesize_T1_Weighted(T1_map, T2_map, PD_map,s):
    TE = 13
    TR = 500
    alpha = np.deg2rad(160)

    tf.cast(TE, dtype=tf.float64)
    tf.cast(TR, dtype=tf.float64)
    tf.cast(alpha, dtype=tf.float64)

    im_tmp = PD_map * (tf.math.sin(alpha) * (1 - tf.math.exp(-TR / T1_map))) / (1 - (tf.math.cos(alpha) * tf.math.exp(-TR / T1_map))) * tf.math.exp(-TE / T2_map)
    im_tmp = tf.math.abs(im_tmp)
    im_syn_T1w = s*im_tmp

    return im_syn_T1w

def synthesize_PD_Weighted(T1_map, T2_map, PD_map,s):
    TE = 12
    TR = 4000

    tf.cast(TE, dtype=tf.float64)
    tf.cast(TR, dtype=tf.float64)

    im_tmp = PD_map * (1 - tf.math.exp(-TR / T1_map)) * tf.math.exp(-TE / T2_map)
    im_syn_PDw = s*im_tmp

    return im_syn_PDw

def synthesize_FLAIR(T1_map, T2_map, PD_map,s):
    TE = 85
    TR = 8000
    TI = 2500
    alpha = np.deg2rad(180)

    tf.cast(TE, dtype=tf.float64)
    tf.cast(TR, dtype=tf.float64)
    tf.cast(TI, dtype=tf.float64)
    tf.cast(alpha, dtype=tf.float64)

    im_tmp  = PD_map * (1- 2*tf.math.exp(-TI/T1_map) + tf.math.exp(-TR/T1_map))/(1+tf.math.exp(-TR/T1_map)*np.cos(alpha))*tf.math.exp(-TE/T2_map);
    im_syn_FLAIR = s*im_tmp

    return im_syn_FLAIR


def contrast_generation(T1_map, T2_map, PD_map,scales):
    T1_map = tf.cast(T1_map, dtype=tf.float64)
    T2_map = tf.cast(T2_map, dtype=tf.float64)
    PD_map = tf.cast(PD_map, dtype=tf.float64)

    T1w =  synthesize_T1_Weighted(T1_map, T2_map, PD_map, scales[0])
    T2w = synthesize_T2_Weighted(T1_map, T2_map, PD_map, scales[1])
    PDw = synthesize_PD_Weighted(T1_map, T2_map, PD_map, scales[2])
    FLAIR = synthesize_FLAIR(T1_map, T2_map, PD_map, scales[3])

    weighted_images = tf.stack((T1w, T2w, PDw, FLAIR), axis=-1)

    return weighted_images


def self_supervised_loss(y_true, y_pred, mask, scale_variables):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    scale_variables = tf.cast(scale_variables, tf.float64)

    pred_T1, pred_T2, pred_PD = y_pred[:, :, :, 0]*5000, y_pred[:, :, :, 1]*500, y_pred[:, :, :, 2]

    pred_images = contrast_generation(pred_T1, pred_T2, pred_PD, scale_variables)

    pred_images_masked = pred_images * mask[:, :, :, np.newaxis]
    y_true_masked = y_true * mask[:, :, :, np.newaxis]

    self_sup_loss = MeanSquaredError()(y_true_masked, pred_images_masked)

    return self_sup_loss