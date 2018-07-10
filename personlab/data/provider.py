import numpy as np
import itertools, random, queue, threading
import pickle, os, fnmatch
from personlab import config
from personlab.data.maker import make_personlab_input



def get_files(provider):
    file_list = os.listdir(config.INPUT_DIR)
    random.shuffle(file_list)
    for file in itertools.cycle(file_list):
        if not provider.is_running:
            break
        if not fnmatch.fnmatch(file, '*.mydat'):
            continue
        try:
            with open(config.INPUT_DIR + file, 'rb') as f:
                dset = pickle.load(f)
        except Exception as e:
            print('FILE: ', file)
            print(e)
            continue
        frames = dset['f']
        seg = dset['seg']
        kp_list = dset['kp']
        seg_all = np.expand_dims(seg.any(axis=1), axis=-1)
        num_frame = len(frames)

        if np.sum(seg_all) / num_frame < config.TAR_H * config.TAR_W / 10: # HUMAN SEG AREA LOWERBOUND
            continue
        provider.request(num_frame, frames, seg_all, kp_list, seg)

class Provider:
    def __init__(self, seeder_func, sampling_type='iter_all'):
        self.q = queue.Queue(maxsize=config.IN_QUEUE_SIZE)
        self.out_q = queue.Queue(maxsize=config.OUT_QUEUE_SIZE)
        self.num_worker_threads = 4
        self.is_running = True
        self.threads = []
        for i in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)
        self.seeder_t = threading.Thread(target=seeder_func, args=(self,))
        self.seeder_t.start()
        self.sampling_type = sampling_type

    def worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            num_frame, frames, seg_all, kp_list, seg = item
            res = make_personlab_input(num_frame, kp_list, seg)
            self.out_q.put((num_frame, frames, seg_all) + res)
            self.q.task_done()
            
    def load(self):
        while True:
            item = self.out_q.get()
            if item is None:
                break
            num_frame, frames, seg_all, hm, so_x, so_y, mo_x, mo_y, lo_x, lo_y = item
            self.out_q.task_done()
            if self.sampling_type == 'iter_all':
                s_i = 0
                while (s_i < num_frame):
                    e_i = min(s_i + config.NUM_FRAME, num_frame)
                    seq_len = e_i - s_i
                    res = frames, seg_all, hm, so_x, so_y, mo_x, mo_y, lo_x, lo_y
                    res = [np.pad(x[s_i:e_i], (0, config.NUM_FRAME - seq_len), 'constant') for x in res]
                    res = tuple(res + [seq_len])
                    yield res
                    s_i = e_i
            elif self.sampling_type == 'random':
                s_i = random.randint(0, num_frame)
                e_i = min(s_i + config.NUM_FRAME, num_frame)
                seq_len = e_i - s_i
                res = frames, seg_all, hm, so_x, so_y, mo_x, mo_y, lo_x, lo_y
                res = [np.pad(res[s_i:e_i], (0, config.NUM_FRAME - seq_len), 'constant') for x in res]
                res = tuple(res + [seq_len])
                yield res

    def request(self, num_frame, frames, seg_all, kp_list, seg):
        self.q.put((num_frame, frames, seg_all, kp_list, seg))

    def close(self):
        self.is_running = False
        self.q.join()
        for i in range(num_worker_threads):
            self.q.put(None)
        for t in self.threads:
            t.join()
