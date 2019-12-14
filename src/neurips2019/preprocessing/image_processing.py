import numpy as np

def calculate_difference_image(img1, img2, tofloat=True):
    diff = img2 - img1

    newmax = 1 if tofloat else 255
    newmin = 0
    out = np.zeros(diff.shape)
    oldmin = diff.reshape(-1,3).min(axis=0)
    oldmax = diff.reshape(-1,3).max(axis=0)

    for dim in range(diff.shape[-1]):
        pxs = diff[:,:,dim]
        oldmin = pxs.min()
        oldmax = pxs.max()
        out[:,:,dim] = (pxs - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin

    return out