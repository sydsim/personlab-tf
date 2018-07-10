import numpy as np



SURREAL_H = 240
SURREAL_W = 320

NUM_KP = 14
NUM_EDGE = 28
MAX_FRAME_SIZE = 10

KP_SEG_LB = 5
AREA_LB = 1000

SURREAL_SEG_NAMES = [
    'lower-belly',  'right-thigh',    'left-thigh',     # 0, 1, 2
    'middle-belly', 'right-shin',     'left-shin',      # 3, 4, 5
    'upper-belly',  'right-foot',     'left-foot',      # 6, 7, 8
    'chest',        'right-toes',     'left-toes',      # 9, 10, 11
    'neck',         'right-shoulder', 'left-shoulder',  # 12, 13, 14
    'face',         'right-upperarm', 'left-upperarm',  # 15, 16, 17
    'right-forearm', 'left-forearm',   # 18, 19
    'right-hand',    'left-hand',      # 20, 21
    'right-fingers', 'left-fingers',   # 22, 23
]

KP_MAP = [
    15, #턱 0
    12, #목 1
    9,  #가슴 2
    16, 18, 20, #오른쪽 어깨, 팔꿈치, 손목 3 4 5
    17, 19, 21, #왼쪽  어깨, 팔꿈치, 손목 6 7 8
    1, 4, 7, #오른쪽 엉덩이, 무릎, 발목 10 11 12
    2, 5, 8, #왼쪽 엉덩이, 무릎, 발목 13 14 15
]
KP_MAP = [
    ('seg', 15), ('seg', 9),
    ('seg', 13), ('seg', 14),
    ('info', (18, 16)), ('info', (19, 17)),
    ('info', (20, 18)), ('info', (21, 19)),
    ('info', ( 1,  0)), ('info', ( 2,  0)),
    ('info', ( 4,  1)), ('info', ( 5,  2)),
    ('info', ( 7,  4)), ('info', ( 8,  5)),
]

KP_NAMES = [
    'head', 'chest',
    'right-shoulder', 'left-shoulder',
    'right-elbow', 'left-elbow',
    'right-wrist', 'left-wrist',
    'right-hip', 'left-hip',
    'right-knee', 'left-knee',
    'right-ankle', 'left-ankle',
]

EDGES = np.array([
    (0, 1),
    (1, 2), (1, 3),
    (2, 4), (3, 5),
    (2, 8), (3, 9),
    (4, 6), (5, 7),
    (8, 9),
    (8, 10), (9, 11),
    (10, 12), (11, 13),
])

EDGES = np.concatenate([EDGES, EDGES[:, ::-1]], axis=0)
