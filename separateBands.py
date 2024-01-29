import numpy as np

def separate_bands(IrawFFT, phaOff, bands, fac):
    phaPerBand = (bands * 2) - 1
    phases = np.array([(2 * np.pi * (p - 1)) / phaPerBand + phaOff for p in range(1, phaPerBand + 1)])
    separate = separate_bands_final(IrawFFT, phases, bands, fac)
    return separate

def separate_bands_final(IrawFFT, phases, bands, fac):
    for i in range(2, bands + 1):
        fac[i - 1] *= 0.5
    
    comp = np.zeros(bands * 2 - 1)
    comp[0] = 0
    for i in range(2, bands + 1):
        comp[(i - 1) * 2 - 1] = i - 1
        comp[(i - 1) * 2] = -(i - 1)
    
    compfac = np.zeros(bands * 2 - 1)
    compfac[0] = fac[0]
    for i in range(2, bands + 1):
        compfac[(i - 1) * 2 - 1] = fac[i - 1]
        compfac[(i - 1) * 2] = fac[i - 1]

    W = np.exp(1j * np.outer(phases, comp))
    for i in range(bands * 2 - 1):
        W[i, :] *= compfac[i]

    siz = IrawFFT.shape[:2]
    S = np.reshape(IrawFFT, (-1, phases.size)) @ np.linalg.pinv(W).T
    Sk = np.reshape(S, siz + (bands * 2 - 1,))
    return Sk
