import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import line_aa
from . import config



def overlay(hm, color=None):
    H, W = hm.shape
    if color is None:
        color = np.random.rand(3)
    return np.tile(color, [H, W, 1]) * np.expand_dims(hm, axis=-1)


def show_heatmap(img, hm, alpha=0.5):
    res = img / 255
    if len(hm.shape) > 2:
        H, W, K = hm.shape
        ov = np.zeros(img.shape)
        for i in range(K):
            ov += overlay(hm[..., i])
        ov = np.clip(ov, 0, 1)
    else:
        ov = overlay(hm)
    res = res * alpha + ov * (1 - alpha)
    return res


def offset_summary(img, hm, hm_true, off_x, off_y, alpha=0.4, stride=10):
    res = img / 255
    if len(hm.shape) > 2:
        H, W, K = hm.shape
        ov = np.zeros(img.shape)
        for i in range(K):
            ov += overlay(hm[..., i])
        ov = np.clip(ov, 0, 1)
    else:
        ov = overlay(hm)
    res = res * alpha + ov * (1 - alpha)
    if len(hm_true.shape) > 2:
        H, W, K = hm_true.shape
        for k in range(K):
            color = np.random.rand(3)
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    if hm_true[y, x, k]:
                        yy, xx, val = line_aa(y, x, int(y+off_y[y, x, k]), int(x+off_x[y, x, k]))
                        yy = np.clip(yy, 0, H-1)
                        xx = np.clip(xx, 0, W-1)
                        val = np.expand_dims(val, axis=-1)
                        res[yy, xx] = val * color + (1 - val) * res[yy, xx]
    else:
        H, W = hm_true.shape
        color = np.random.rand(3)
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if hm_true[y, x]:
                    yy, xx, val = line_aa(y, x, int(y+off_y[y, x]), int(x+off_x[y, x]))
                    yy = np.clip(yy, 0, H-1)
                    xx = np.clip(xx, 0, W-1)
                    val = np.expand_dims(val, axis=-1)
                    res[yy, xx] = val * color + (1 - val) * res[yy, xx]

    return res.astype(np.float32)

def summary_skeleton(img, kp_map):
    res = img / 255
    H, W, C = img.shape

    num_people = kp_map[..., 0].max()
    for p_i in range(num_people):
        color = np.random.rand(3)
        kp_point = np.zeros([config.NUM_KP, 3], dtype=np.int16)
        for y, x in zip(*np.nonzero(kp_map[..., 0] == p_i + 1)):
            kp_i = kp_map[y, x, 1]
            kp_point[kp_i-1] = (x, y, 1)

        for x, y, c in kp_point:
            if c == 0:
                continue
            start = (y-5, x-5)
            end = (y+5, x+5)
            res[start[0]:end[0], start[1]:end[1]] = color

        for e1, e2 in config.EDGES:
            if kp_point[e1, 2] == 0 or kp_point[e2, 2] == 0:
                continue
            sx, sy, _ = kp_point[e1]
            ex, ey, _ = kp_point[e2]
            yy, xx, val = line_aa(sy, sx, ey, ex)
            yy = np.clip(yy, 0, H-1)
            xx = np.clip(xx, 0, W-1)
            val = np.expand_dims(val, axis=-1)
            res[yy, xx] = val * color + (1 - val) * res[yy, xx]

    return res.astype(np.float32)
