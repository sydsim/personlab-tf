import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from personlab import config
from personlab.util import *



def model_base(model_output, inner_h, inner_w):
    batch_size = config.BATCH_SIZE
    num_kp = config.NUM_KP
    num_edge = config.NUM_EDGE
    target_h = config.TAR_H
    target_w = config.TAR_W

    hm   = slim.conv2d(model_output, num_kp,   [1, 1], activation_fn=tf.sigmoid)
    seg  = slim.conv2d(model_output, 1,        [1, 1], activation_fn=tf.sigmoid)
    so_x = slim.conv2d(model_output, num_kp,   [1, 1], activation_fn=None)
    so_y = slim.conv2d(model_output, num_kp,   [1, 1], activation_fn=None)
    mo_x = slim.conv2d(model_output, num_edge, [1, 1], activation_fn=None)
    mo_y = slim.conv2d(model_output, num_edge, [1, 1], activation_fn=None)
    lo_x = slim.conv2d(model_output, num_kp,   [1, 1], activation_fn=None)
    lo_y = slim.conv2d(model_output, num_kp,   [1, 1], activation_fn=None)

    b, y, x, _ = np.indices([batch_size, inner_h, inner_w, num_edge])
    i = np.tile(config.EDGES[:, 0], [batch_size, inner_h, inner_w, 1])
    for _ in range(config.NUM_RECURRENT):
        mo_p = [b, y + mo_y, x + mo_x, i]
        mo_x = gather_bilinear(so_x, mo_p, (inner_h, inner_w)) + mo_x
        mo_y = gather_bilinear(so_y, mo_p, (inner_h, inner_w)) + mo_y

    b, y, x, i = np.indices([batch_size, inner_h, inner_w, num_kp])
    for _ in range(config.NUM_RECURRENT_2):
        lo_p = [b, y + lo_y, x + lo_x, i]
        lo_x = gather_bilinear(lo_x, lo_p, (inner_h, inner_w)) + lo_x
        lo_y = gather_bilinear(lo_y, lo_p, (inner_h, inner_w)) + lo_y

    for _ in range(config.NUM_RECURRENT_3):
        lo_p = [b, y + lo_y, x + lo_x, i]
        lo_x = gather_bilinear(so_x, lo_p, (inner_h, inner_w)) + lo_x
        lo_y = gather_bilinear(so_y, lo_p, (inner_h, inner_w)) + lo_y

    split_size = [num_kp, 1, num_kp, num_kp, num_edge, num_edge, num_kp, num_kp]
    pred_sum = tf.concat([hm, seg, so_x, so_y, mo_x, mo_y, lo_x, lo_y], axis=-1)
    pred_sum = tf.image.resize_images(pred_sum, (target_h, target_w), method=tf.image.ResizeMethod.BILINEAR)
    hm, seg, so_x, so_y, mo_x, mo_y, lo_x, lo_y = tf.split(pred_sum, split_size, axis=-1)
    so_x, mo_x, lo_x = [x / (inner_w - 1) * (target_w - 1) for x in [so_x, mo_x, lo_x]]
    so_y, mo_y, lo_y = [y / (inner_h - 1) * (target_h - 1) for y in [so_y, mo_y, lo_y]]

    return hm, seg, so_x, so_y, mo_x, mo_y, lo_x, lo_y
