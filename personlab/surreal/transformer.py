import random
import numpy as np
import cv2 as cv
from personlab import config
from . import const as surreal

class Transformer:
    def __init__(self, original_h, original_w, target_h, target_w):
        self.target_h, self.target_w = target_h, target_w
        self.original_h, self.original_w = original_h, original_w
        self.R = config.RADIUS
        self.X = np.tile(np.arange(-self.R, self.R + 1), [2 * self.R + 1, 1])
        self.Y = self.X.transpose()
        self.M = np.sqrt(self.X * self.X + self.Y * self.Y) <= self.R

    def get_cover(self, cx, cy):
        x_off_s = max(self.R - cx, 0)
        x_off_e = max(min(self.R, config.TAR_W - cx - 1) + self.R + 1, 0)
        y_off_s = max(self.R - cy, 0)
        y_off_e = max(min(self.R, config.TAR_H - cy - 1) + self.R + 1, 0)
        ym = self.Y[y_off_s:y_off_e, x_off_s:x_off_e]
        xm = self.X[y_off_s:y_off_e, x_off_s:x_off_e]
        mm = self.M[y_off_s:y_off_e, x_off_s:x_off_e]
        y_i = (ym + cy).astype(np.int32)
        x_i = (xm + cx).astype(np.int32)
        return x_i[mm], y_i[mm]

    def build(self, fraction=None, transform_offset=None, is_flip=False):
        f = max(self.target_w / self.original_w, self.target_h / self.original_h)
        if fraction:
            f = max(fraction, f)
        if transform_offset:
            offset_h, offset_w = transform_offset
        else:
            offset_h = random.uniform(0, max(0, self.original_h * f - self.target_h)) // 2
            offset_w = random.uniform(0, max(0, self.original_w * f - self.target_w)) // 2
        mat = np.array([
            [1., 0., self.target_w * f // 2 - offset_w],
            [0., 1., self.target_h * f // 2 - offset_h],
            [0., 0., 1.]]) # transform back to (0, 0)
        if is_flip:
            mat = mat.dot(np.array(
                [[-1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])) # flip
        mat = mat.dot(np.array(
            [[f, 0., 0.],
            [0., f, 0.],
            [0., 0., 1.]])) # scale
        mat = mat.dot(np.array(
            [[1., 0., -self.original_w // 2],
            [0., 1., -self.original_h // 2],
            [0., 0., 1.]])) # transform to center
        return mat


    def run(self, frames, kp, kp_idx, kp_seg,
            fraction=None, transform_offset=None, is_flip=False):
        num_frame = frames.shape[0]
        mat = self.build(fraction, transform_offset, is_flip)
        frames = [cv.warpAffine(frames[i], mat[0:2], (self.target_h, self.target_w)) for i in range(num_frame)]
        frames = np.stack(frames, axis=0)

        kp_ex = kp[..., 2].copy()
        kp[..., 2] = 1
        kp = kp.dot(mat.transpose())

        hm = np.zeros([num_frame, config.TAR_H, config.TAR_W, surreal.NUM_KP]).astype(np.uint8)
        so_x = np.zeros([num_frame, config.TAR_H, config.TAR_W, surreal.NUM_KP]).astype(np.int16)
        so_y = np.zeros([num_frame, config.TAR_H, config.TAR_W, surreal.NUM_KP]).astype(np.int16)
        mo_x = np.zeros([num_frame, config.TAR_H, config.TAR_W, surreal.NUM_EDGE]).astype(np.int16)
        mo_y = np.zeros([num_frame, config.TAR_H, config.TAR_W, surreal.NUM_EDGE]).astype(np.int16)


        for f_i in range(num_frame):
            seg = kp_seg[f_i, ...]
            seg_resize = cv.warpAffine(seg, mat[0:2], (self.target_h, self.target_w))

            kp_exist = kp_ex[f_i, :]
            for kp_i in range(surreal.NUM_KP):
                if not kp_exist[kp_i]:
                    continue
                cx = int(round(kp[f_i, kp_i, 0]))
                cy = int(round(kp[f_i, kp_i, 1]))
                x_i, y_i = self.get_cover(cx, cy)
                for s_i in kp_idx[f_i, kp_i]:
                    seg_i = seg_resize == s_i
                    if np.sum(seg_i[y_i, x_i]) < surreal.KP_SEG_LB:
                        kp_exist[kp_i] = False
                if kp_exist[kp_i]:
                    hm[f_i, y_i, x_i, kp_i] = 1
                    so_x[f_i, y_i, x_i, kp_i] = cx - x_i
                    so_y[f_i, y_i, x_i, kp_i] = cy - y_i
                else:
                    kp_ex[f_i, kp_i] = 0

            for e_i in range(surreal.NUM_EDGE):
                k_i = surreal.EDGES[e_i, 0]
                k_j = surreal.EDGES[e_i, 1]
                if not kp_exist[k_i] or not kp_exist[k_j]:
                    continue
                cx = int(round(kp[f_i, k_i, 0]))
                cy = int(round(kp[f_i, k_i, 1]))
                x_i, y_i = self.get_cover(cx, cy)
                mo_x[f_i, y_i, x_i, e_i] = kp[f_i, k_j, 0] - x_i
                mo_y[f_i, y_i, x_i, e_i] = kp[f_i, k_j, 1] - y_i

        kp[..., 2] = kp_ex
        return (frames, hm, so_x, so_y, mo_x, mo_y, kp)
