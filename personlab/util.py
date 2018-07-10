import os
import tensorflow as tf

def get_file_list(path):
    res = []
    for fn in os.listdir(path):
        if not os.path.isfile(path + fn):
            res += get_file_list(path + fn + '/')
        elif fn.endswith('mp4'):
            name = fn.split('.')[0]
            res.append(path + name)
    return res


def bilinear(indices, shape):
    H, W = shape[:2]
    oy = tf.clip_by_value(indices[1], 0, H-1-1e-8)
    ox = tf.clip_by_value(indices[2], 0, W-1-1e-8)
    iy = [tf.floor(oy), tf.ceil(oy + 1e-9)]
    ix = [tf.floor(ox), tf.ceil(ox + 1e-9)]
    idx_p = []
    for y in iy:
        for x in ix:
            indices[1] = y
            indices[2] = x
            idx = tf.cast(tf.stack(indices, axis=-1), tf.int32)
            p = (1 - tf.abs(y - oy)) * (1 - tf.abs(x - ox))
            idx_p.append((idx, p))
    return idx_p

def gather_bilinear(params, indices, shape):
    idx_p = bilinear(indices, shape)
    res = []
    for idx, p in idx_p:
        r = tf.gather_nd(params, idx)
        res.append(r * p)
    return tf.add_n(res)

def scatter_bilinear(params, indices, shape):
    idx_p = bilinear(indices)
    res = []
    for idx, p in idx_p:
        r = tf.scatter_nd(idx, params, shape)
        if len(r.shape) > len(p.shape):
            p = tf.expand_dims(p, axis=-1)
        res.append(r * p)
    return tf.add_n(res)

def resize(tensor, shape):
    return tf.image.resize_images(
        tensor,
        shape,
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)
