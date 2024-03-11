import numpy as np
import time
from separateBands import separate_bands
from sim_otf_provider import sim_otf_provider
from importimages import import_images
from NfourierShift import Nfourier_shift
from place_Freq import place_freq
from SVD_spectrum import svd_spectrum

# 初始化参数
filter_size = 11
mask_size = 3
PCA_num = 1
dist = 0.15
overlap = dist
divideByOtf = True

start_time = time.time()

for angle_num in range(1, param['nrDirs'] + 1):
    param['fac'] = [1, 1]
    param['phaOff'] = 0
    spectrum = separate_bands(IIrawFFT[:, :, (angle_num - 1) * param['nrPhases']:angle_num * param['nrPhases']], param['phaOff'], param['nrBands'], param['fac'])
    temp = spectrum[:, :, 1] * NotchFilter  # NotchFilter需要定义
    yPos, xPos = np.unravel_index(np.argmax(temp), temp.shape)

    peak = {'xPos': xPos + 1, 'yPos': yPos + 1}

    for pca_size in range(1, PCA_num + 1):
        if pca_size == 1:
            kx = (peak['xPos'] - Center[1])  # Center需要定义
            ky = (peak['yPos'] - Center[0])
            old_kx = kx
            old_ky = ky
        else:
            kx = old_kx
            ky = old_ky

        temp = Nfourier_shift(place_freq(spectrum[:, :, 1]), -(2 - 1) * old_kx, -(2 - 1) * old_ky)
        MASK = np.ones(temp.shape)
        NPixel = temp.shape[0] // 2  # Assuming square images for simplicity
        MASK[NPixel-mask_size:NPixel+mask_size+1, NPixel-mask_size:NPixel+mask_size+1] = 0
        temp[MASK == 1] = 0
        ROI = temp[NPixel-filter_size:NPixel+filter_size+1, NPixel-filter_size:NPixel+filter_size+1]


        NEW_Phase_ROI, dx, dy = svd_spectrum(ROI, filter_size)

        old_kx += dx
        old_ky += dy


        param['Dir'][angle_num-1] = {'px': old_kx, 'py': old_ky}

        pca_x[angle_num-1] = old_kx
        pca_y[angle_num-1] = old_ky

        param['Dir'][angle_num-1]['phaOff'] = -np.angle(NEW_Phase_ROI[filter_size, filter_size])

        K0[angle_num-1] = np.sqrt(old_kx**2 + old_ky**2)

        