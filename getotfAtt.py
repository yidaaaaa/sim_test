import numpy as np

def val_attenuation(dist, fwhm):
    """
    Calculates the attenuation value based on the distance and the full width at half maximum (FWHM).

    Parameters:
    dist (ndarray): The distance matrix.
    fwhm (float): The full width at half maximum value.

    Returns:
    ndarray: The calculated attenuation values.
    """
    return (1 - np.exp(-np.power(dist, 4) / (2 * np.power(fwhm, 4))))

def get_otf_att(img_size, cycles_per_micron, att_fwhm, kx, ky):
    """
    Generates an Optical Transfer Function (OTF) attenuation matrix.

    Parameters:
    img_size (int): The size of the image (assuming a square image).
    cycles_per_micron (float): Cycles per micron.
    att_fwhm (float): Full width at half maximum for the attenuation.
    kx (float): The x-coordinate shift.
    ky (float): The y-coordinate shift.

    Returns:
    ndarray: The OTF attenuation matrix.
    """
    w = img_size
    h = img_size
    siz = np.array([h, w])
    cnt = siz // 2 + 1
    kx += cnt[1]
    ky += cnt[0]
    otf_att = np.zeros((h, w))
    y, x = np.meshgrid(np.arange(1, w+1), np.arange(1, h+1))
    rad = np.hypot(y - ky, x - kx)
    cycl = rad * cycles_per_micron
    otf_att = val_attenuation(cycl, att_fwhm)

    return otf_att