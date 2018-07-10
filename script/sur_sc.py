import numpy as np
from personlab.datamaker.surreal import surreal
from personlab.datamaker.maker import make_input
import pickle, queue, threading
from pathlib import Path

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        path, filename = item
        res = surreal.read_surreal(path)
        if res is None:
            continue
        num_frame, frame_res, seg_res, kp_list, com = res
        dset = {
            'f': frame_res,
            'seg': seg_res,
            'kp': kp_list
        }
        with open('tr/' + filename, 'wb') as f:
            pickle.dump(dset, f)
        q.task_done()

num_worker_threads = 8
q = queue.Queue()
threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

base = '/home/ubuntu/surreal/train/'
cnt = 0
for path in surreal.get_file_list(base):
    fname = path.split('/')[-1]
    filename = 'surreal_%d_%s.mydat' % (cnt, fname)

    cnt += 1
    my_file = Path('tr/' + filename)
    if my_file.exists():
        print(filename, 'exist')
        continue
    q.put((path, filename))
q.join()
for i in range(num_worker_threads):
    q.put(None)
for t in threads:
    t.join() 

