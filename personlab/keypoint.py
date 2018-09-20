import numpy as np
from personlab import config
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter, maximum_filter


def get_keypoints(hm, so_x, so_y):
    y_i, x_i, kp_i = np.indices(hm.shape)
    ox = x_i + so_x
    oy = y_i + so_y
    iy = [oy.astype(np.int16), oy.astype(np.int16) + 1]
    ix = [ox.astype(np.int16), ox.astype(np.int16) + 1]
    p_list = []
    for y in iy:
        for x in ix:
            p = hm * (1 - np.abs(y - oy)) * (1 - np.abs(x - ox))
            p_list.append((p, y, x))
    p_s, y_s, x_s = [np.stack(l, axis=0) for l in zip(*p_list)]

    b_y = np.logical_and(y_s >= 0, y_s < config.TAR_H)
    b_x = np.logical_and(x_s >= 0, x_s < config.TAR_W)
    b_s = np.logical_and(b_y, b_x)

    kp_candidates = []
    for i in range(config.NUM_KP):
        p, x, y, b = [d[..., i] for d in (p_s, x_s, y_s, b_s)]
        h = coo_matrix((p[b], (y[b], x[b])), shape=[config.TAR_H, config.TAR_W]).todense()
        h_gf = gaussian_filter(h, sigma=2)
        h_mf = np.logical_and(maximum_filter(h_gf, size=5) == h_gf, h_gf > config.KP_PEAK_LB)
        kp_cands = []
        for y, x in zip(*np.nonzero(h_mf)):
            kp_cands.append((h_gf[y, x], y, x, i))
        kp_candidates += kp_cands
    kp_candidates.sort(reverse=True)
    return kp_candidates


def get_avg(kp_x, kp_y, mo_x, mo_y, radius):
    y_i, x_i = np.indices([config.TAR_H, config.TAR_W])
    to_y = kp_y - y_i
    to_x = kp_x - x_i
    in_r = to_y ** 2 + to_x ** 2 < radius ** 2
    in_r_size = np.sum(in_r)
    tar_y = np.sum(to_y[in_r] + mo_y[in_r]) / in_r_size + kp_y
    tar_x = np.sum(to_x[in_r] + mo_x[in_r]) / in_r_size + kp_x
    return (tar_x, tar_y)


def check_adjacency(p_i, kp_list, include_in):
    _, cy, cx, _ = kp_list[p_i]
    for p_j, dat in enumerate(kp_list):
        if include_in[p_j] is None:
            continue
        _, py, px, _ = dat
        dist = ((py - cy) ** 2 + (px - cx) ** 2) ** 0.5
        if dist < config.KP_DUPLICATE_DIST:
            include_in[p_i] = -1
            return True
    return False


def construct_keypoint_map(hm, so_x, so_y, mo_x, mo_y):
    kp_list = get_keypoints(hm, so_x, so_y)
    person_cnt = 0
    include_in = [None] * len(kp_list)
    for p_i in range(len(kp_list)):
        if include_in[p_i] is not None or check_adjacency(p_i, kp_list, include_in):
            continue

        person_cnt += 1
        person_id = person_cnt
        include_in[p_i] = person_id
        q = [p_i]
        while (len(q) > 0):
            p_cur = q.pop()
            _, cy, cx, ck = kp_list[p_cur]
            for e_i, nodes in enumerate(config.EDGES):
                e1, e2 = nodes
                if e1 != ck:
                    continue

                tx, ty = get_avg(cx, cy, mo_x[..., e_i], mo_y[..., e_i], 5)
                min_dist = 1e9
                max_hval = 0
                min_p = None
                for p_j, dat in enumerate(kp_list):
                    hval, py, px, pk = dat
                    if pk != e2 or include_in[p_j] is not None or check_adjacency(p_j, kp_list, include_in):
                        continue
                    dist = ((py - ty) ** 2 + (px - tx) ** 2) ** 0.5
                    if dist < config.KP_LINK_DIST and min_dist > dist:
                        max_hval = hval
                        min_dist = dist
                        min_p = p_j
                if min_p is not None:
                    _, py, px, pk = kp_list[min_p]
                    include_in[min_p] = person_id
                    q.append(min_p)
    kp_map = np.zeros([config.TAR_H, config.TAR_W, 2], dtype=np.int16)
    for p_i, dat in enumerate(kp_list):
        if include_in[p_i] is None or include_in[p_i] == -1:
            continue
        _, py, px, pk = kp_list[p_i]
        kp_map[py, px] = (include_in[p_i], pk + 1)
    return kp_map

