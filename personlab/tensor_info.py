import tensorflow as tf
from . import config

INPUT_TENSOR_INFO = [
    {
        'name': 'image',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, 3),
        'type': tf.uint8,
    },{
        'name': 'hm',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.bool,
    },{
        'name': 'seg',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, 1),
        'type': tf.bool,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.int16,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.int16,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_EDGE),
        'type': tf.int16,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_EDGE),
        'type': tf.int16,
    },{
        'name': 'lo_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.int16,
    },{
        'name': 'lo_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.int16,
    },{
        'name': 'kp_map',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, 2),
        'type': tf.uint8,
    }
]
