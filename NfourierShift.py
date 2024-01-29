import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def fourier_shift(vec, kx, ky):
    h, w = vec.shape
    y, x = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')
    x = x / w
    y = y / h
    comshift = vec * np.exp(2 * np.pi * 1j * (ky * y + kx * x))
    return comshift

def Nfourier_shift(inv, kx, ky):
    inv = ifft2(fftshift(inv))
    outv = fourier_shift(inv, kx, ky)
    outv = fftshift(fft2(outv))
    return outv