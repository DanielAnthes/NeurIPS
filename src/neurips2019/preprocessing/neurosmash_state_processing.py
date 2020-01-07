import os
import numpy as np
import cv2
from datetime import datetime

STATES_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'states/')


def state_to_screen(state, size=None, outsize=580, tofloat=True, norm=False, gray=False):
    """
    Takes the state representation returned from the Neurosmash envrionment and forms it into
    a square image only containing the rotated platform.
    
    If tofloat is true output will be in [0,1] otherwise in [0,255] with results being integers.
    If norm is true, image will be normalised before returning resulting in a gray background and colored agents.
    If gray is true, image will be turned into grayscale before returning.
    """
    if not size:
        size = np.int(np.sqrt(len(state)/3))

    scaler = lambda x: _scale_to_int(x, size, 768)
    screen = np.reshape(state, (size, size, 3)).astype(np.uint8)

    if tofloat:
        screen = screen.astype(np.float64)
        screen /= 256

    bordersize = scaler(69)
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
        [scaler(384) + bordersize, bordersize],
        [size + bordersize, scaler(244) + bordersize],
        [scaler(384) + bordersize, scaler(545) + bordersize],
        [bordersize, scaler(244) + bordersize]
    ])
    pts2 = np.float32([
        [scaler(580) - scaler(20), scaler(22)],
        [scaler(580) - scaler(71), scaler(580) - scaler(64)],
        [0, scaler(580)],
        [scaler(65), scaler(65)]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    rect = scaler(580)
    screen = cv2.warpPerspective(screen, M, (rect, rect))
    screen = cv2.resize(screen, (outsize, outsize))

    if gray:
        screen = rgb2gray(screen)
    if norm:
        screen = normalize(screen)

    return screen


def normalize(picture, newmin=0, newmax=255):
    """
    Normalizes an image
    """
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
    """
    Reutrns a grayscale version of the image with a focus on boosting the first and thirds channel (usually red & blue)
    """
    dtype = img.dtype
    img = img.astype(np.float64)
    img[:, :, 0] *= 0.4  # 0.2989
    img[:, :, 1] *= 0.35  # 0.5870
    img[:, :, 2] *= 0.25  # 0.1140
    return img.mean(axis=2).astype(dtype)


def _scale_to_int(num, nsize, ref):
    """
    Scales a number according to a scale and reference value
    """
    return int(num * (nsize / ref) + 0.5)


def save_states(states, agent_name, savedir=STATES_SAVEDIR):
    """
    Saves states to a a given directory
    """
    print(savedir)
    states = np.array(states, dtype=np.int8)
    filename = f"states_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    print(filename)
    with open(os.path.join(savedir, filename), 'wb') as fp:
        print(fp)
        np.save(fp, states)
