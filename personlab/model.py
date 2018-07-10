import functools, operator, copy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from personlab.surreal.feeder import *
from personlab import config
from personlab.surreal import const
from personlab.display import *

def train(model_func, log_dir, type='single'):
    tf.reset_default_graph()

    surreal_feeder = SurrealFeeder()
    if type == 'single':
        tensors = surreal_feeder.single_tensors()
    else:
        tensors = surreal_feeder.multi_tensors()

    res, init_func = model_func(tensors)
    hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred = res

    hm_loss = - tf.reduce_mean(tensors['hm'] * tf.log(hm_pred + 1e-9) + (1 - tensors['hm']) * tf.log(1 - hm_pred + 1e-9))
    so_loss = tf.abs(tensors['so_x'] - so_x_pred) / config.RADIUS + tf.abs(tensors['so_y'] - so_y_pred) / config.RADIUS
    mo_loss = tf.abs(tensors['mo_x'] - mo_x_pred) / config.RADIUS + tf.abs(tensors['mo_y'] - mo_y_pred) / config.RADIUS

    disc_only = tf.cast(tensors['hm'], tf.float32)
    disc_size = tf.reduce_sum(disc_only, axis=[1, 2]) + 1e-9
    so_loss = tf.reduce_mean(tf.reduce_sum(so_loss * disc_only, axis=[1, 2]) / disc_size)

    disc_only = tf.cast(tf.gather(tensors['hm'], const.EDGES[:, 0], axis=-1), tf.float32)
    disc_size = tf.reduce_sum(disc_only, axis=[1, 2]) + 1e-9
    mo_loss = tf.reduce_mean(tf.reduce_sum(mo_loss * disc_only, axis=[1, 2]) / disc_size)

    total_loss = hm_loss * 4.0 + so_loss * 1.0 + mo_loss * 0.5

    # summary
    b_i, kp_i = 0, 0
    args = [tensors['image'][b_i], hm_pred[b_i], tensors['hm'][b_i], mo_x_pred[b_i], mo_y_pred[b_i]]
    image_tensor1 = tf.py_func(summary_offset, args, tf.float32)
    image_tensor1.set_shape([None, None, 4])
    tf.summary.image('sum', tf.expand_dims(image_tensor1, 0))
    tf.summary.scalar('losses/hm_loss', hm_loss)
    tf.summary.scalar('losses/so_loss', so_loss)
    tf.summary.scalar('losses/mo_loss', mo_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    tf.contrib.slim.learning.train(train_op,
                                   log_dir,
                                   init_fn=init_func,
                                   log_every_n_steps=100,
                                   save_summaries_secs=60,
                                   session_config=sess_config,
                                  )
