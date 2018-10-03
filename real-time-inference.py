
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from personlab import display, config
from personlab.models.model_base import model_base
from personlab.nets.mobilenet import mobilenet_v2
from personlab.keypoint import construct_keypoint_map
import cv2
import glob
import numpy as np
import time
import os

images_path = glob.glob('your_path_here')
out_path = 'your_path_here'

input_feed_shape = (1, config.TAR_W, config.TAR_H, 3)
img_tf = tf.placeholder(dtype=tf.float32, shape=input_feed_shape)

with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        frame_tensor = tf.cast(img_tf, tf.float32)
        model_output, _ = mobilenet_v2.mobilenet_base(frame_tensor, output_stride=config.STRIDE)

inner_h = int(config.TAR_H - 1) // config.STRIDE + 1
inner_w = int(config.TAR_W - 1) // config.STRIDE + 1
res = model_base(model_output, inner_h, inner_w)

variables = slim.get_model_variables()
restore_map = {}

for v in variables:
    org_name = v.name.split(':')[0]
    restore_map[org_name] = v

checkpoint_path = './logs/model.ckpt-86162'

init_assign_func = slim.d(checkpoint_path, restore_map)

sess = tf.Session()
init_assign_func(sess)

def main():
	for im_path in images_path:
		
		img = cv2.resize( cv2.imread(im_path), (config.TAR_W, config.TAR_H) )
		start_time = time.time()
		img = np.expand_dims(img, axis = 0)
		preds = sess.run(res, feed_dict={img_tf: img} )
		print('FPS: ', 1 / (time.time() - start_time))


		hm, seg, so_x, so_y, mo_x, mo_y, lo_x, lo_y = preds
		kp_map_pred = [construct_keypoint_map(_hm, _so_x, _so_y, _mo_x, _mo_y) \
		              for _hm, _so_x, _so_y, _mo_x, _mo_y in zip(hm, so_x, so_y, mo_x, mo_y)]

		img = np.concatenate(img, axis=0)
		seg = np.concatenate(seg, axis=0)
		kp_map_pred = np.concatenate(kp_map_pred, axis=0)

		kp_map = display.summary_skeleton(img, kp_map_pred) * 255
		mask = display.show_heatmap(img, seg) * 255

		
		combined = np.concatenate((mask, kp_map), axis=1)
		#cv2.imshow('tf-pose-estimation result', kp_map)
		#cv2.waitKey(5)
		cv2.imwrite(os.path.join(out_path, im_path.split('/')[-1]), combined)
	
if __name__ == '__main__':
    main()