import numpy as np

def fade_border_cos(img, px):
    # h, w = img.shape
    h = img.shape[0]
    w = img.shape[1]
    fac = 1 / px * np.pi / 2
    dat = np.copy(img)
    
    # top and bottom
    for y in range(px):
        dat[y, :] *= np.power(np.sin(y * fac), 2)
        dat[-(y+1), :] *= np.power(np.sin(y * fac), 2)
    
    # left and right
    for x in range(px):
        dat[:, x] *= np.power(np.sin(x * fac), 2)
        dat[:, -(x+1)] *= np.power(np.sin(x * fac), 2)

    return dat

def import_images(images):
    N = images.shape[2]
    L = images.shape[0]
    error_num = images.shape[3]
    for i in range(N):
        cur_img = images[:, :, i, :]
        if L > 256:
            cur_img = fade_border_cos(cur_img, 10)
        else:
            cur_img = fade_border_cos(cur_img, 0)
        images[:, :, i, :] = cur_img
    # for i in range(N):
    #     cur_img = images[:, :, i]
    #     if L > 256:
    #         cur_img = fade_border_cos(cur_img, 10)
    #     else:
    #         cur_img = fade_border_cos(cur_img, 0)
    #     images[:, :, i] = cur_img
    return images


def import_images1(images):
    if len(images.shape) == 3:
        N = images.shape[2]
    elif len(images.shape) == 2:
        N = 1
    L = images.shape[0]

    for i in range(N):
        if len(images.shape) == 3:
            cur_img = images[:, :, i]
        elif len(images.shape) == 2:
            cur_img = images
        if L > 256:
            cur_img = fade_border_cos(cur_img, 10)
        else:
            cur_img = fade_border_cos(cur_img, 0)
        if len(images.shape) == 3:
            images[:, :, i] = cur_img
        elif len(images.shape) == 2:
            images = cur_img
    return images
