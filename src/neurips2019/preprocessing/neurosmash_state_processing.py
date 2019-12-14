import numpy as np
import cv2


def process(state, size=768, outsize=580, tofloat=True, normalize=True):
    """
    Currently only works for size=768! size and outsize should as such should not be changed!
    """
    screen = np.reshape(state, (size, size, 3)).astype(np.uint8)

    if tofloat:
        screen = screen.astype(np.float64)
        screen /= 256
    
    rows, cols, ch = screen.shape

    bordersize = 69
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
        [384+bordersize, bordersize],
        [size+bordersize, 244+bordersize],
        [384+bordersize, 545+bordersize],
        [bordersize, 244+bordersize]
    ])
    pts2 = np.float32([
        [580-20, 22],
        [580-71, 580-64],
        [0, 580],
        [65, 65]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    screen = cv2.warpPerspective(screen, M, (580, 580))

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
