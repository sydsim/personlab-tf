import numpy as np
from numpy import newaxis as nX
from personlab import config
from scipy.sparse import coo_matrix



H = config.TAR_H
W = config.TAR_W
R = config.RADIUS
shape = (H, W)

Y, X = np.indices(shape)
Y_r, X_r = np.indices([2*R+1, 2*R+1]) - R
Y_r = Y_r[nX, ...]
X_r = X_r[nX, ...]
B_r = Y_r * Y_r + X_r * X_r < R * R


def make_disk(cx, cy, valid, mx, my):
    y = cy[:, nX, nX] + Y_r
    x = cx[:, nX, nX] + X_r
    y_b = np.logical_and(y >= 0, y < H)
    x_b = np.logical_and(x >= 0, x < W)
    valid = np.logical_and(valid[:, nX, nX], B_r)
    b = np.logical_and(valid, np.logical_and(y_b, x_b))
    y_val = my[:, nX, nX] - Y_r
    x_val = mx[:, nX, nX] - X_r

    disk = np.zeros(shape, dtype=np.bool)
    y_off = np.zeros(shape, dtype=np.int16)
    x_off = np.zeros(shape, dtype=np.int16)
    disk[y[b], x[b]] = True
    y_off[y[b], x[b]] = y_val[b]
    x_off[y[b], x[b]] = x_val[b]
    return disk, x_off, y_off


def make_long_offset(tx, ty, valid, seg):
    y = ty[:, nX, nX] - Y[nX, ...]
    x = tx[:, nX, nX] - X[nX, ...]
    seg_val = np.logical_and(valid[:, nX, nX], seg)
    y_t = np.concatenate([Y[s] for s in seg_val])
    x_t = np.concatenate([X[s] for s in seg_val])
    y_off = coo_matrix((y[seg_val], (y_t, x_t)), shape=shape).todense()
    x_off = coo_matrix((x[seg_val], (y_t, x_t)), shape=shape).todense()
    return x_off, y_off


def construct_personlab_input(kp_list, seg):
    hm = np.zeros([H, W, config.NUM_KP]).astype(np.uint8)
    so_x = np.zeros([H, W, config.NUM_KP]).astype(np.int16)
    so_y = np.zeros([H, W, config.NUM_KP]).astype(np.int16)
    mo_x = np.zeros([H, W, config.NUM_EDGE]).astype(np.int16)
    mo_y = np.zeros([H, W, config.NUM_EDGE]).astype(np.int16)
    lo_x = np.zeros([H, W, config.NUM_KP]).astype(np.int16)
    lo_y = np.zeros([H, W, config.NUM_KP]).astype(np.int16)
    kp_map = np.zeros([H, W, 2]).astype(np.uint8)

    num_people = kp_list.shape[2]

    for kp_i in range(config.NUM_KP):
        cx, cy, cb = kp_list[kp_i]
        disk, x_off, y_off = make_disk(cx, cy, cb > 0, np.zeros(cb.shape), np.zeros(cb.shape))
        hm[..., kp_i] = disk
        so_x[..., kp_i] = x_off
        so_y[..., kp_i] = y_off
        x_off, y_off = make_long_offset(cx, cy, cb > 0, seg)
        lo_x[..., kp_i] = x_off
        lo_y[..., kp_i] = y_off
        for p_i in range(num_people):
            if 0 <= cx[p_i] and cx[p_i] < config.TAR_W and \
                0 <= cy[p_i] and cy[p_i] < config.TAR_H and cb[p_i] > 0:
                kp_map[cy[p_i], cx[p_i]] = (p_i+1, kp_i+1)

    for e_i, e in enumerate(config.EDGES):
        k_1, k_2 = e
        x1, y1, b1 = kp_list[k_1]
        x2, y2, b2 = kp_list[k_2]
        b = np.logical_and(b1 > 0, b2 > 0)
        _, x_off, y_off = make_disk(x1, y1, b, x2-x1, y2-y1)
        mo_x[..., e_i] = x_off
        mo_y[..., e_i] = y_off

    return hm, so_x, so_y, mo_x, mo_y, lo_x, lo_y, kp_map
