import time, itertools, random
from threading import Thread
from queue import Queue

import numpy as np
import cv2 as cv
import scipy.io as sio

from .. import config
from .transformer import Transformer
from .const import *
from personlab.util import get_file_list
from personlab.surreal.tensor_info import *


class SurrealFeeder:
    def __init__(self, num_workers=1, is_single=True, in_queue_size=5, out_queue_size=5):
        self.num_workers = num_workers
        self.path_list = get_file_list(config.TRAIN_DATA_BASE_DIR)
        self.in_queue = Queue(maxsize=in_queue_size)
        self.out_queue = Queue(maxsize=out_queue_size)
        self.master_thread = Thread(target=self._distribute)
        self.worker_threads = [Thread(target=self._process) for _ in range(num_workers)]
        self.transformer = Transformer(SURREAL_H, SURREAL_W, config.TAR_H, config.TAR_W)


    def start(self):
        self.is_running = True
        self.master_thread.start()
        for worker_thread in self.worker_threads:
            worker_thread.start()

    def single_sync(self):
        random.shuffle(self.path_list)
        for path in self.path_list: #itertools.cycle(self.path_list):
            res = self.read_surreal(path)
            if res is None:
                continue
            frames, kp_list, kp_idx, kp_seg = res
            res = self.transformer.run(frames, kp_list, kp_idx, kp_seg)
            img, hm, so_x, so_y, mo_x, mo_y, kp = res
            num_frame = len(img)
            for i in range(num_frame):
                if np.sum(hm[i, ...]) < AREA_LB:
                    continue
                yield img[i], hm[i], so_x[i], so_y[i], mo_x[i], mo_y[i], kp[i]

    def multi_sync(self):
        random.shuffle(self.path_list)
        for path in self.path_list: #itertools.cycle(self.path_list):
            res = self.read_surreal(path)
            if res is None:
                continue
            frames, kp_list, kp_idx, kp_seg = res
            res = self.transformer.run(frames, kp_list, kp_idx, kp_seg)
            img, hm, so_x, so_y, mo_x, mo_y, kp = res
            num_frame = len(img)
            for i in range(num_frame):
                if np.sum(hm[i, ...]) < AREA_LB:
                    continue
                yield img[i], hm[i], so_x[i], so_y[i], mo_x[i], mo_y[i], kp[i]

    def stop(self, graceful=False):
        if graceful:
            for _ in range(self.num_workers):
                self.in_queue.put(None)
            self.in_queue.join()

        self.is_running = False
        for worker_thread in self.worker_threads:
            worker_thread.join()
        self.master_thread.join()


    def _distribute(self):
        for path in itertools.cycle(self.path_list):
            if not self.is_running:
                break
            res = self.read_surreal(path)
            if res is None:
                continue
            self.in_queue.put(res)
            time.sleep(0)
        print('master stopped!')


    def read_surreal(self, path):
        cap = cv.VideoCapture(path + '.mp4')
        frames = []
        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        frames = np.array(frames).astype(np.uint8)
        num_frame = len(frames)
        if num_frame < MAX_FRAME_SIZE:
            return None

        info = sio.loadmat(path + '_info.mat')
        seg = sio.loadmat(path + '_segm.mat')
        kp_list = np.zeros([num_frame, NUM_KP, 3]).astype(np.int16)
        kp_idx = np.zeros([num_frame, NUM_KP, 2]).astype(np.uint8)
        kp_seg = np.zeros([num_frame, SURREAL_H, SURREAL_W]).astype(np.uint8)

        Y, X = np.mgrid[:SURREAL_H, :SURREAL_W]
        for f_i in range(num_frame):
            f_seg = seg['segm_%d' % (f_i+1)]
            kp_seg[f_i] = f_seg
            for i, kp_info in enumerate(KP_MAP):
                t_name, t_i = kp_info
                if t_name == 'seg' and np.sum(f_seg == t_i+1) > 0:
                    y = int(np.average(Y[f_seg == t_i+1]))
                    x = int(np.average(X[f_seg == t_i+1]))
                    kp_list[f_i, i] = (x, y, 1)
                    kp_idx[f_i, i] = [t_i+1, t_i+1]
                elif t_name == 'info':
                    t_i1, t_i2 = t_i
                    x = info['joints2D'][0, t_i1, f_i]
                    y = info['joints2D'][1, t_i1, f_i]
                    kp_list[f_i, i] = (x, y, 1)
                    kp_idx[f_i, i] = [t_i1+1, t_i2+1]
        return frames, kp_list, kp_idx, kp_seg


    def _process(self):
        print('worker thread start!')
        while self.is_running:
            item = self.in_queue.get()
            if item is None:
                self.in_quee.task_done()
                break
            frames, kp_list, kp_idx, kp_seg = item
            res = self.transformer.run(frames, kp_list, kp_idx, kp_seg)
            self.out_queue.put(res)
            self.in_queue.task_done()
            time.sleep(0)
        print('worker stopped!')


    def single_retrive(self):
        while self.is_running:
            item = self.out_queue.get()
            if item is None:
                self.out_queue.task_done()
                break
            img, hm, so_x, so_y, mo_x, mo_y = item
            num_frame = len(img)
            for i in range(num_frame):
                if np.sum(hm[i, ...]) < AREA_LB:
                    continue
                yield img[i], hm[i], so_x[i], so_y[i], mo_x[i], mo_y[i]
            self.out_queue.task_done()


    def multi_retrive(self):
        while self.is_running:
            item = self.out_queue.get()
            if item is None:
                self.out_queue.task_done()
                break
            img, hm, so_x, so_y, mo_x, mo_y, num_frames = item
            for i in range(num_frame // MAX_FRAME_SIZE):
                f_i_s = i * MAX_FRAME_SIZE
                f_i_e = f_i_s + MAX_FRAME_SIZE
                kp_part = kp_list[f_i_s:f_i_e]
                if np.count_nonzero(kp_part[...,2]) < KP_LB * MAX_FRAME_SIZE:
                    continue

                img, hm, so_x, so_y, mo_x, mo_y, num_frames = transform(frames[f_i_s:f_i_e,...], kp_part)
                if np.sum(hm) < AREA_LB * MAX_FRAME_SIZE:
                    continue
                yield img, hm, so_x, so_y, mo_x, mo_y, num_frames
            self.out_queue.task_done()

    def _get_tensors(self, feed_func, tensor_info):
        types = tuple(t['type'] for t in tensor_info)
        input_tensors = tf.data.Dataset.from_generator(feed_func, types) \
                            .batch(config.BATCH_SIZE) \
                            .prefetch(config.PREFETCH_SIZE) \
                            .make_one_shot_iterator() \
                            .get_next()
        tensors = {}
        for tensor, info in zip(input_tensors, tensor_info):
            tensor.set_shape(info['shape'])
            tensors[info['name']] = tensor
        return tensors


    def single_tensors(self):
        return self._get_tensors(self.single_sync, INPUT_SINGLE_TENSOR_INFO)

    def multi_tensors(self):
        return self._get_tensors(self.multi_sync, INPUT_TENSOR_INFO)
