import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import coo_matrix
from skimage.draw import line_aa
from . import config



def bilinear(_indices, shape):
    H, W = shape[:2]
    indices = _indices.copy()
    EPS = 1e-7
    oy = tf.clip_by_value(indices[1], 0, H - 1 - EPS)
    ox = tf.clip_by_value(indices[2], 0, W - 1 - EPS)
    iy = [tf.floor(oy), tf.floor(oy) + 1]
    ix = [tf.floor(ox), tf.floor(ox) + 1]
    idx_p = []
    for y in iy:
        for x in ix:
            indices[1] = y
            indices[2] = x
            idx = tf.cast(tf.stack(indices, axis=-1), tf.int32)
            p = (1 - tf.abs(y - oy)) * (1 - tf.abs(x - ox))
            idx_p.append((idx, p))
    idx, p = [tf.stack(t, axis=0) for t in zip(*idx_p)]
    return idx, p


def gather_bilinear(params, indices, shape):
    idx, p = bilinear(indices, shape)
    r = tf.gather_nd(params, idx)
    return tf.reduce_sum(r * p, axis=0)


def scatter_bilinear(params, indices, shape):
    idx, p = bilinear(indices, shape)
    r = tf.scatter_nd(params, idx, shape)
    return tf.reduce_sum(r * p, axis=0)
