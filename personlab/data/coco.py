import queue, threading, random
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from personlab import config
from personlab.data.datautil import construct_personlab_input
from scipy.misc import imresize



TAR_H = config.TAR_H
TAR_W = config.TAR_W


class CocoDataGenerator:
    def __init__(self, base_dir, inst_json, person_json):
        self.base_dir = base_dir
        self.coco_inst = COCO(inst_json)
        self.coco_kps = COCO(person_json)
        self.img_ids = self.coco_inst.getImgIds(catIds=[1])
        self.img_infos = self.coco_inst.loadImgs(self.img_ids)
        self.num_workers = config.WORKER_SIZE
        self.in_q = queue.Queue(maxsize=config.IN_QUEUE_SIZE)
        self.out_q = queue.Queue(maxsize=config.OUT_QUEUE_SIZE)


    def read_coco(self, img_info):
        img = io.imread(self.base_dir + img_info['file_name'])
        annIds = self.coco_kps.getAnnIds(imgIds=img_info['id'], catIds=[1], iscrowd=None)
        anns = self.coco_kps.loadAnns(annIds)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]
            img = img * [1, 1, 1]

        org_h, org_w, _ = img.shape
        min_size = min(org_h, org_w)
        cr_h = min_size
        cr_w = min_size
        sy = random.randint(0, org_h - cr_h)
        sx = random.randint(0, org_w - cr_w)
        ey = sy + cr_h
        ex = sx + cr_w

        num_person = len(anns)
        image_res = imresize(img[sy:ey, sx:ex, :], size=(TAR_H, TAR_W), interp='bilinear')
        seg_res = np.zeros([num_person, TAR_H, TAR_W], dtype=np.uint8)
        kp_list = np.zeros([config.NUM_KP, 3, num_person], dtype=np.int16)

        for p_i, ann in enumerate(anns):
            seg_org = self.coco_kps.annToMask(ann)
            seg_res[p_i] = imresize(seg_org[sy:ey, sx:ex], size=(TAR_H, TAR_W), interp='nearest')
            kp_coco = np.array([ann['keypoints']]).reshape([-1, 3])
            cx = kp_coco[:, 0]
            cy = kp_coco[:, 1]
            cb = kp_coco[:, 2]
            cx = (cx - sx) / cr_w * TAR_W
            cy = (cy - sy) / cr_h * TAR_H
            cb = cb > 0
            kp_list[:, 0, p_i] = cx
            kp_list[:, 1, p_i] = cy
            kp_list[:, 2, p_i] = cb

        seg_all = np.expand_dims(seg_res.any(axis=0), axis=-1)
        return image_res, seg_res, seg_all, kp_list


    def provider(self):
        for img_info in self.img_infos:
            self.in_q.put(img_info)
        for _ in range(self.num_workers):
            self.in_q.put(None)
        self.in_q.join()
        self.out_q.put(None)


    def loader(self):
        worker_threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            worker_threads.append(t)
        provider_thread = threading.Thread(target=self.provider)
        provider_thread.start()

        while True:
            input_data = self.out_q.get()
            self.out_q.task_done()
            if input_data is None:
                break
            yield input_data

        self.out_q.join()
        for t in worker_threads:
            t.join()
        provider_thread.join()


    def worker(self):
        while True:
            img_info = self.in_q.get()
            self.in_q.task_done()
            if img_info is None:
                break
            image, seg, seg_all, kp_list = self.read_coco(img_info)
            hm, so_x, so_y, mo_x, mo_y, lo_x, lo_y, kp_map = construct_personlab_input(kp_list, seg)
            input_data = (image, hm, seg_all, so_x, so_y, mo_x, mo_y, lo_x, lo_y, kp_map)
            self.out_q.put(input_data)
