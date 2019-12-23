import numpy as np

def calculate_difference_image(img1, img2, tofloat=True):
    """
    calculated and returns the difference image between img1 and img2,
    where img2 is regarded as the later image time-wise and as such img1 is subtracted from img2
    """
    diff = img2 - img1
    if len(diff.shape) == 2:
        diff = diff.reshape((diff.shape[0], diff.shape[1], 1))
    channels = diff.shape[-1]

    newmax = 1 if tofloat else 255
    newmin = 0
    out = np.zeros(diff.shape)
    oldmin = diff.reshape(-1,channels).min(axis=0)
    oldmax = diff.reshape(-1,channels).max(axis=0)

    for dim in range(channels):
        pxs = diff[:,:,dim]
        oldmin = pxs.min()
        oldmax = pxs.max()
        out[:,:,dim] = (pxs - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin

    if channels == 1:
        out = out.squeeze()

    return out
