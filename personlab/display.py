from matplotlib import pyplot as plt
from skimage.draw import line_aa
from personlab import config as surreal
import numpy as np
from . import config


def overlay(hm, color=None):
    H, W = hm.shape
    if color is None:
        color = np.random.rand(3)
    return np.tile(color, [H, W, 1]) * np.expand_dims(hm, axis=-1)

def show_heatmap(img, hm, alpha=0.5):
    plt.rcParams['figure.figsize'] = [25, 25]
    plt.figure()

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
    plt.imshow(res)

def show_offset(img, p, off_x, off_y, stride=10):
    plt.rcParams['figure.figsize'] = [25, 25]
    plt.figure()
    H, W, K = off_x.shape

    res = img / 255

    offsets = []
    for k in range(K):
        color = np.random.rand(3)
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if p[y, x, k]:
                    plt.arrow(x, y, off_x[y, x, k], off_y[y, x, k], width=0.1, color=color)
    return plt.imshow(res)

def summary_offset(img, hm, hm_true, off_x, off_y, alpha=0.4, stride=10):
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

def rectangle(start, end, shape):
    hs, ws = np.round(start).astype(np.int32)
    he, we = np.round(end).astype(np.int32)
    hs = max(min(hs, shape[0]), 0)
    he = max(min(he, shape[0]), 0)
    ws = max(min(ws, shape[1]), 0)
    we = max(min(we, shape[1]), 0)
    return np.mgrid[hs:he, ws:we]

def summary_skeleton(img, kp):
    res = img / 255
    H, W, C = img.shape
    color = np.random.rand(3)
    for x, y, c in kp:
        if c == 0:
            continue
        start = (y-5, x-5)
        end = (y+5, x+5)
        rr, cc = rectangle(start, end, shape=res.shape)
        res[rr, cc] = color
    color = np.random.rand(3)
    for kp_i, kp_j in surreal.EDGES:
        sx, sy, c1 = np.round(kp[kp_i]).astype(np.int32)
        ex, ey, c2 = np.round(kp[kp_j]).astype(np.int32)
        if c1 == 0 or c2 == 0:
            continue
        yy, xx, val = line_aa(sy, sx, ey, ex)
        yy = np.clip(yy, 0, H-1)
        xx = np.clip(xx, 0, W-1)
        val = np.expand_dims(val, axis=-1)
        res[yy, xx] = val * color + (1 - val) * res[yy, xx]
    return res.astype(np.float32)
