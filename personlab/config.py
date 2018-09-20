import numpy as np



TAR_H = 401
TAR_W = 401

WORKER_SIZE = 8
IN_QUEUE_SIZE = 20
OUT_QUEUE_SIZE = 20

BATCH_SIZE = 5
PREFETCH_SIZE = 20
SHUFFLE_BUFFER_SIZE = 100
NUM_EPOCH = 10

NUM_RECURRENT = 2
NUM_RECURRENT_2 = 2
NUM_RECURRENT_3 = 2

RADIUS = 32
STRIDE = 8

KP_PEAK_LB = 5
KP_LINK_DIST = 16
KP_DUPLICATE_DIST = 8

KP_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]
NUM_KP = len(KP_NAMES)

EDGES = np.array([
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [ 5, 11],
    [ 6, 12],
    [ 5,  6],
    [ 5,  7],
    [ 6,  8],
    [ 7,  9],
    [ 8, 10],
    [ 1,  2],
    [ 0,  1],
    [ 0,  2],
    [ 1,  3],
    [ 2,  4],
    [ 3,  5],
    [ 4,  6]
])
EDGES = np.concatenate([EDGES, EDGES[:, ::-1]], axis=0) # bidirectional
NUM_EDGE = len(EDGES)
