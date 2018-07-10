import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from personlab import config
from personlab.surreal import const as surreal
from personlab.util import *
from personlab.nets import resnet_v2, resnet_utils

def resnet_single(tensors):
    MD_H = int((config.TAR_H-1)//config.STRIDE)+1
    MD_W = int((config.TAR_W-1)//config.STRIDE)+1

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        _, end_points = resnet_v2.resnet_v2_101(tensors['image'], output_stride=config.STRIDE)
        model_output = end_points['resnet_v2_101/block4']
        hm_pred = slim.conv2d(model_output, surreal.NUM_KP, [1, 1], activation_fn=tf.sigmoid)
        so_x_pred = slim.conv2d(model_output, surreal.NUM_KP, [1, 1], activation_fn=None)
        so_y_pred = slim.conv2d(model_output, surreal.NUM_KP, [1, 1], activation_fn=None)
        mo_x_pred = slim.conv2d(model_output, surreal.NUM_EDGE, [1, 1], activation_fn=None)
        mo_y_pred = slim.conv2d(model_output, surreal.NUM_EDGE, [1, 1], activation_fn=None)

        b, y, x, _ = np.mgrid[:config.BATCH_SIZE, :MD_H, :MD_W, :surreal.NUM_EDGE]
        i = np.tile(surreal.EDGES[:, 0], [config.BATCH_SIZE, MD_H, MD_W, 1])
        for _ in range(config.NUM_RECURRENT):
            mo_p = [b, y+mo_y_pred, x+mo_x_pred, i]
            mo_x_pred = gather_bilinear(so_x_pred, mo_p, (MD_H, MD_W)) + mo_x_pred
            mo_y_pred = gather_bilinear(so_y_pred, mo_p, (MD_H, MD_W)) + mo_y_pred
        hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred = [resize(x, (config.TAR_H, config.TAR_W)) for x in [hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred]]
        so_x_pred, so_y_pred, mo_x_pred, mo_y_pred = [x * config.STRIDE for x in [so_x_pred, so_y_pred, mo_x_pred, mo_y_pred]]
        res = hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred

    checkpoint_path = 'pretrained/resnet/resnet_v2_101.ckpt'
    variables = slim.get_model_variables()
    restore_map = {}
    for v in variables:
        if not v.name.startswith('resnet'):
            continue
        org_name = v.name.split(':')[0]
        restore_map[org_name] = v
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, restore_map)
    def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

    return (res, InitAssignFn)
