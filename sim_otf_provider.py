import numpy as np
from functions import write_otf_vector
from numpy.fft import ifftn, fftshift

def val_ideal_OTF(dist):
    if dist < 0 or dist > 1:
        return 0
    return (1 / np.pi) * (2 * np.arccos(dist) - np.sin(2 * np.arccos(dist)))

def val_attenuation(dist, strength, fwhm):
    return 1 - strength * (np.exp(-np.power(dist, 2) / (np.power(0.5 * fwhm, 2))) ** 1)

def from_estimate(ret):
    ret['isMultiband'] = 0
    ret['isEstimate'] = 1
    vals1 = np.zeros(ret['sampleLateral'])
    valsAtt = np.zeros(ret['sampleLateral'])
    valsOnlyAtt = np.zeros(ret['sampleLateral'])

    for i in range(ret['sampleLateral']):
        v = abs(i) / ret['sampleLateral']
        r1 = val_ideal_OTF(v) * np.power(ret['estimateAValue'], v)
        vals1[i] = r1

    for i in range(ret['sampleLateral']):
        dist = abs(i) * ret['cyclesPerMicron']
        valsOnlyAtt[i] = val_attenuation(dist, ret['attStrength'], ret['attFWHM'])
        valsAtt[i] = vals1[i] * valsOnlyAtt[i]

    ret['vals'] = vals1
    ret['valsAtt'] = valsAtt
    ret['valsOnlyAtt'] = valsOnlyAtt
    return ret

def get_only_att(ret, kx, ky):
    w = ret['imgSize']
    h = ret['imgSize']
    siz = [h, w]
    cnt = [s // 2 + 1 for s in siz]
    kx = kx + cnt[1]
    ky = ky + cnt[0]
    onlyatt = np.zeros((h, w))

    y, x = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    rad = np.hypot(y - ky, x - kx)
    cycl = rad * ret['cyclesPerMicron']
    onlyatt = val_attenuation(cycl, ret['attStrength'], ret['attFWHM'])
    return onlyatt


def sim_otf_provider(param, NA, lambda_, a):
    ret = {
        'na': NA,
        'lambda': lambda_,
        'cutoff': 1000 / (0.5 * lambda_ / NA),
        'imgSize': param['imgSize'],
        'cyclesPerMicron': param['cyclesPerMicron'],
        'sampleLateral': int(np.ceil(1000 / (0.5 * lambda_ / NA) / param['cyclesPerMicron']) + 1),
        'estimateAValue': a,
        'maxBand': 2,
        'attStrength': param['attStrength'],
        'attFWHM': 1.0,
        'useAttenuation': 1
    }

    ret = from_estimate(ret)
    ret['otf'] = np.zeros((param['imgSize'], param['imgSize']))
    ret['otfatt'] = np.zeros((param['imgSize'], param['imgSize']))
    ret['onlyatt'] = np.zeros((param['imgSize'], param['imgSize']))

    ret['otf'] = write_otf_vector(ret['otf'], ret, 1, 0, 0)
    ret['onlyatt'] = get_only_att(ret, 0, 0)
    ret['otfatt'] = ret['otf'] * ret['onlyatt']

    return ret


def otf2psf(otf):
    """
    Convert an Optical Transfer Function (OTF) to a Point Spread Function (PSF).

    Parameters:
    otf (numpy.ndarray): The Optical Transfer Function.

    Returns:
    numpy.ndarray: The Point Spread Function.
    """
    psf = ifftn(otf)

    psf = np.abs(fftshift(psf))

    return psf