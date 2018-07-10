import tensorflow as tf
from .. import config
from . import const as surreal

INPUT_TENSOR_INFO = [
    {
        'name': 'image',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, 3),
        'type': tf.float32,
    },{
        'name': 'hm',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'seq_len',
        'shape': (config.BATCH_SIZE,),
        'type': tf.int32,
    }
]

OUTPUT_TENSOR_INFO = [
    {
        'name': 'hm',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.int32,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, surreal.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    }
]

INPUT_SINGLE_TENSOR_INFO = [
    {
        'name': 'image',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, 3),
        'type': tf.float32,
    },{
        'name': 'hm',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'kp',
        'shape': (config.BATCH_SIZE, surreal.NUM_KP, 3),
        'type': tf.float32,
    },
]

OUTPUT_SINGLE_TENSOR_INFO = [
    {
        'name': 'hm',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.int32,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, config.TAR_H, config.TAR_W, surreal.NUM_EDGE),
        'type': tf.float32,
    }
]
