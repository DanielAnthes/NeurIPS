import os
import numpy as np
import cv2
import pickle
from datetime import datetime

STATES_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'states/')

def process(state, size=768, outsize=580, tofloat=True, normalize=True):
    """
    Currently only works for size=768! size and outsize should as such should not be changed!
    """
    screen = np.reshape(state, (size, size, 3)).astype(np.uint8)

    if tofloat:
        screen = screen.astype(np.float64)
        screen /= 256

    bordersize = _scale_to_int(69, size, 768)
    bordercolor = [0, 0, 0]

    screen = cv2.copyMakeBorder(
        screen,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=bordercolor
    )

    pts1 = np.float32([
        [_scale_to_int(384, size, 768)+bordersize, bordersize],
        [size+bordersize, _scale_to_int(244, size, 768)+bordersize],
        [_scale_to_int(384, size, 768)+bordersize, _scale_to_int(545, size, 768)+bordersize],
        [bordersize, _scale_to_int(244, size, 768)+bordersize]
    ])
    pts2 = np.float32([
        [_scale_to_int(580, size, 768)-_scale_to_int(20, size, 768), _scale_to_int(22, size, 768)],
        [_scale_to_int(580, size, 768)-_scale_to_int(71, size, 768), _scale_to_int(580, size, 768)-_scale_to_int(64, size, 768)],
        [0, _scale_to_int(580, size, 768)],
        [_scale_to_int(65, size, 768), _scale_to_int(65, size, 768)]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    screen = cv2.warpPerspective(screen, M, (outsize, outsize))

    if normalize:
        screen = normalize((screen))

    return screen

def normalize(picture, newmin=0, newmax=255):
    #     dst = np.zeros(picture.shape)
    #     return cv2.normalize(picture, dst=dst, alpha=newmin, beta=newmax, norm_type=cv2.NORM_INF)
    if len(picture.shape) == 2:
        oldmin = picture.min()
        oldmax = picture.max()
        out = (picture - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    elif len(picture.shape) == 3:
        n, m, d = picture.shape
        s = picture.sum(axis=2).reshape(n, m, 1)
        s[s == 0] = 1
        out = (picture / s * 255).astype(np.uint8)
    return out


def rgb2gray(img):
    dtype = img.dtype
    img = img.astype(np.float64)
    img[:, :, 0] *= 0.4  # 0.2989
    img[:, :, 1] *= 0.35  # 0.5870
    img[:, :, 2] *= 0.25  # 0.1140
    return img.mean(axis=2).astype(dtype)

def _scale_to_int(num, nsize, ref):
    return int(num * nsize/ref + 0.5)


def save_states(states, agent_name, savedir = STATES_SAVEDIR):
    print(savedir)
    states = np.array(states, dtype=np.int8)
    filename = f"states_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    print(filename)
    with open(os.path.join(savedir, filename), 'wb') as fp:
        print(fp)
        np.save(fp, states)