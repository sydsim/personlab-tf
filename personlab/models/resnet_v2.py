import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from personlab import config
from personlab.models.model_base import model_base
from personlab.nets import resnet_v2, resnet_utils
from personlab.preprocessing  import inception_preprocessing



INNER_H = INNER_W = 299
MD_H = MD_W = 38

def resnet_v2_model(image_tensor, resnet_fn, checkpoint_path=None, is_training=False):
    def inception_preproc_fn(image_raw):
        image = inception_preprocessing.preprocess_image(image_raw, INNER_H, INNER_W, \
                                                         is_training=is_training,  \
                                                         add_image_summaries=False)
        return image

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        image_reformed = tf.map_fn(inception_preproc_fn, image_tensor, dtype=tf.float32)
        model_output, _ = resnet_fn(image_reformed, \
                                    global_pool=False, \
                                    num_classes=None, \
                                    output_stride=config.STRIDE, \
                                    is_training=is_training)

    res = model_base(model_output, MD_H, MD_W)
    variables = slim.get_model_variables()
    restore_map = {}
    for v in variables:
        if is_training and not v.name.startswith('resnet'):
            continue
        org_name = v.name.split(':')[0]
        restore_map[org_name] = v
    init_assign_func = slim.assign_from_checkpoint_fn(checkpoint_path, restore_map)
    return (res, init_assign_func)


def resnet_v2_101(tensors, checkpoint_path, is_training):
    return resnet_v2_model(tensors, resnet_v2.resnet_v2_101, checkpoint_path, is_training)


def resnet_v2_152(tensors, checkpoint_path, is_training):
    return resnet_v2_model(tensors, resnet_v2.resnet_v2_152, checkpoint_path, is_training)
