import tensorflow as tf
import tensorflow.contrib.slim as slim
from personlab import config
from personlab.nets.mobilenet import mobilenet_v2
from personlab.models.model_base import model_base



def mobilenet_v2_model(image_tensor, checkpoint_path=None, is_training=False):
    with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
        frame_tensor = tf.cast(image_tensor, tf.float32)
        model_output, _ = mobilenet_v2.mobilenet_base(frame_tensor, output_stride=config.STRIDE)

    inner_h = int(config.TAR_H - 1) // config.STRIDE + 1
    inner_w = int(config.TAR_W - 1) // config.STRIDE + 1
    res = model_base(model_output, inner_h, inner_w)

    variables = slim.get_model_variables()
    restore_map = {}
    for v in variables:
        if is_training and not v.name.startswith('MobilenetV2'):
            continue
        org_name = v.name.split(':')[0]
        restore_map[org_name] = v
    init_assign_func = slim.assign_from_checkpoint_fn(checkpoint_path, restore_map)
    return (res, init_assign_func)
