import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from personlab import config, display
from personlab.keypoint import construct_keypoint_map
from personlab.tensor_info import INPUT_TENSOR_INFO as tensor_info



def train(model_func, data_generator, checkpoint_path, log_dir):
    EPS = 1e-7
    types = tuple(t['type'] for t in tensor_info)
    d = tf.data.Dataset.from_generator(data_generator, output_types=types)
    d = d.batch(config.BATCH_SIZE, drop_remainder=True)
    d = d.shuffle(config.SHUFFLE_BUFFER_SIZE)
    d = d.repeat(config.NUM_EPOCH)
    d = d.prefetch(config.PREFETCH_SIZE)

    input_tensors = d.make_one_shot_iterator().get_next()
    tensors = {}
    for tensor, info in zip(input_tensors, tensor_info):
        tensor.set_shape(info['shape'])
        tensors[info['name']] = tensor

    output, init_func = model_func(tensors['image'], checkpoint_path=checkpoint_path, is_training=True)
    hm_pred, seg_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred, lo_x_pred, lo_y_pred = output
    hm_pred  = tf.clip_by_value(hm_pred,  EPS, 1-EPS)
    seg_pred = tf.clip_by_value(seg_pred, EPS, 1-EPS)

    hm_loss  = tf.reduce_mean(tf.where(tensors['hm'],  -tf.log(hm_pred),  -tf.log(1 - hm_pred)))
    seg_loss = tf.reduce_mean(tf.where(tensors['seg'], -tf.log(seg_pred), -tf.log(1 - seg_pred)))

    def loss_calc_func(ox_true, oy_true, ox_pred, oy_pred, loss_seg):
        ox_true_f = tf.cast(ox_true, tf.float32)
        oy_true_f = tf.cast(oy_true, tf.float32)
        loss_mat = tf.abs(ox_true_f - ox_pred) / config.RADIUS + \
                   tf.abs(oy_true_f - oy_pred) / config.RADIUS
        loss_seg_f = tf.cast(loss_seg, tf.float32)
        loss_seg_size = tf.reduce_sum(loss_seg_f, axis=[1, 2]) + EPS
        loss_in_seg = tf.where(loss_seg, loss_mat, tf.zeros_like(loss_mat))
        loss = tf.reduce_sum(loss_in_seg, axis=[1, 2]) / loss_seg_size
        return tf.reduce_mean(loss)

    so_loss = loss_calc_func(tensors['so_x'], tensors['so_y'], \
                             so_x_pred, so_y_pred, \
                             tensors['hm'])
    mo_loss = loss_calc_func(tensors['mo_x'], tensors['mo_y'], \
                             mo_x_pred, mo_y_pred, \
                             tf.gather(tensors['hm'], config.EDGES[:, 0], axis=-1))
    lo_loss = loss_calc_func(tensors['lo_x'], tensors['lo_y'], \
                             lo_x_pred, lo_y_pred, \
                             tf.tile(tensors['seg'], (1, 1, 1, config.NUM_KP)))

    total_loss = hm_loss * 4.0 + seg_loss * 2.0 + so_loss * 1.0 + mo_loss * 0.25 + lo_loss * 0.125

    b_i = 0
    args = [tensors['image'], hm_pred, tensors['hm'], so_x_pred, so_y_pred]
    args = [tf.cast(x[b_i], tf.float32) for x in args]
    offset_summary = tf.py_func(display.offset_summary, args, tf.float32)
    offset_summary.set_shape([None, None, 4])
    tf.summary.image('sum', tf.expand_dims(offset_summary, 0))
    tf.summary.scalar('losses/hm_loss', hm_loss)
    tf.summary.scalar('losses/seg_loss', seg_loss)
    tf.summary.scalar('losses/so_loss', so_loss)
    tf.summary.scalar('losses/mo_loss', mo_loss)
    tf.summary.scalar('losses/lo_loss', lo_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    tf.contrib.slim.learning.train(train_op,
                                   log_dir,
                                   init_fn=init_func,
                                   log_every_n_steps=100,
                                   save_summaries_secs=300,
                                   session_config=sess_config,
                                  )


def evaluate(model_func, data_generator, checkpoint_path, num_batches=-1):
    types = tuple(t['type'] for t in tensor_info)
    d = tf.data.Dataset.from_generator(data_generator, output_types=types)
    d = d.batch(config.BATCH_SIZE)
    d = d.prefetch(config.PREFETCH_SIZE)

    input_tensors = d.make_one_shot_iterator().get_next()
    tensors = {}
    for tensor, info in zip(input_tensors, tensor_info):
        tensor.set_shape(info['shape'])
        tensors[info['name']] = tensor

    output, init_func = model_func(tensors['image'], checkpoint_path=checkpoint_path, is_training=False)
    result = {'image': [],
              'kp_map_true': [],
              'kp_map_pred': [],
              'seg_true': [],
              'seg_pred': [],
             }
    saver = tf.train.Saver()
    true_keys = [info['name'] for info in tensor_info]

    with tf.Session() as sess:
        init_func(sess)
        try:
            while num_batches != 0:
                num_batches -= 1
                true_tensor = [tensors[name] for name in true_keys]
                trues, preds = sess.run((true_tensor, output))
                tval = {}
                for k, v in zip(true_keys, trues):
                    tval[k] = v

                hm, seg, so_x, so_y, mo_x, mo_y, lo_x, lo_y = preds
                kp_map_pred = [construct_keypoint_map(_hm, _so_x, _so_y, _mo_x, _mo_y) \
                               for _hm, _so_x, _so_y, _mo_x, _mo_y in zip(hm, so_x, so_y, mo_x, mo_y)]

                result['image'].append(tval['image'])
                result['kp_map_true'].append(tval['kp_map'])
                result['kp_map_pred'].append(kp_map_pred)
                result['seg_true'].append(tval['seg'])
                result['seg_pred'].append(seg)

        except tf.errors.OutOfRangeError:
            pass

    result_concat = {}
    for key, val in result.items():
        result_concat[key] = np.concatenate(val, axis=0)

    return result_concat
