import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import fft2, ifft2, fftshift
from scipy import signal
import imageio
from sim_otf_provider import sim_otf_provider, otf2psf
from importimages import import_images, import_images1
from skimage.restoration import richardson_lucy

size_num = 50
pca_x = np.zeros((3, size_num))
pca_y = np.zeros((3, size_num))
cor_x = np.zeros((3, size_num))
cor_y = np.zeros((3, size_num))

frames = imageio.mimread('RawImage/raw.tif', memtest=False)

# Parameters
NA = 0.6 # NA
Mag = 40 # Magnification
Pixelsize = 6.5 / Mag # Pixelszie
wavelength = 561 # Laser wavelength
lambda_ = 0.607 # Fluorescent wavelength
sub_optimization = 0

# load image

Iraw = np.zeros((frames[0].shape[0], frames[0].shape[1], len(frames)))  # 增加一个维度来存储不同噪声水平的图像

# for error_num in range(size_num):
for i, frame in enumerate(frames):
    # 添加高斯白噪声
    noise_level = size_num  # 噪声水平
    noise = np.random.normal(0, noise_level, frame.shape)
    Iraw[:, :, i] = frame + noise

# plt.figure()
# plt.imshow(Iraw[:, :, 1, 1], cmap='gray')
# plt.show()
        
# 显示原始图像的第一帧进行验证
        
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(frames[0], cmap='gray')
# plt.title('Original Frame')

# # 显示经过添加噪声后的第一帧图像
# plt.subplot(1, 2, 2)
# plt.imshow(Iraw[:, :, 0, 30], cmap='gray') 
# plt.title('Noise Added Frame')

# plt.show()

NPixel = Iraw.shape[0]
# Generate approximate OTF/PSF
param = {}

param['imgSize'] = NPixel
param['micronsPerPixel'] = Pixelsize
param['cyclesPerMicron'] = 1 / (NPixel * param['micronsPerPixel'])
param['NA'] = NA
param['lambda'] = lambda_
param['cutoff'] = 1000 / (0.5 * param['lambda'] / param['NA'])
param['sampleLateral'] = int(np.ceil(param['cutoff'] / param['cyclesPerMicron']) + 1)
param['nrBands'] = 2
param['phaOff'] = 0
param['fac'] = np.ones(param['nrBands'])
param['attStrength'] = 0
param['OtfProvider'] = sim_otf_provider(param, param['NA'], param['lambda'], 1)
PSF = np.abs(otf2psf(param['OtfProvider']['otf']))


param['nrDirs'] = 3
param['nrPhases'] = 3
N = param['nrDirs'] * param['nrPhases']
IIraw = import_images1(Iraw)
IIraw_deconv = np.zeros_like(IIraw)
# for error_num in range(size_num):
for i in range(N):
    IIraw_deconv[:, :, i] = richardson_lucy(IIraw[:, :, i], PSF, num_iter=10)

IIrawFFT = np.zeros_like(IIraw_deconv, dtype=np.complex128)
# for error_num in range(size_num):
for i in range(N):
    IIrawFFT[:, :, i] = fft2(IIraw_deconv[:, :, i])

# for i in range(param['nrDirs']):
#     for j in range(param['nrPhases']):
#         IIraw[:, :, (i - 1) * param['nrDirs'] + j, :] = Iraw[:, :, (i - 1) * param['nrDirs'] + j, :] / param['nrPhases']

WF = np.zeros((NPixel, NPixel))
Tdeconv = np.zeros((NPixel, NPixel))
WFdeconv = np.zeros((NPixel, NPixel, param['nrDirs']))
WFdeconvFFT = np.zeros((NPixel, NPixel, param['nrDirs']))

# for error_num in range(size_num):
for i in range(param['nrDirs']):
    for j in range(param['nrPhases']):
        Tdeconv[:, :] = Tdeconv[:, :] + IIraw[:, :, i * param['nrDirs'] + j+1]
        WF[:, :] = WF[:, :] + Iraw[:, :, i * param['nrDirs'] + j+1]
    WFdeconv[:, :, i] = Tdeconv[:, :] / param['nrPhases']
    WFdeconvFFT[:, :, i] = fft2(WFdeconv[:, :, i])

# WF = WF / N
# WF = np.abs(fft2(WF))
# for error_num in range(size_num):
# WF[:, :] = np.abs(fft2(WF[:, :])) / N
WF = WF / N
WF = import_images1(WF)

# matlab code: figure(),imshow(WF,[]),colormap hot;

plt.figure()
plt.imshow(WF[:, :, 0], cmap='hot')
plt.show()
    

fftWF2 = np.zeros((2*NPixel, 2*NPixel))
# for error_num in range(size_num):
fftWF2[NPixel/2 +1:3*NPixel/2, NPixel/2 +1:3*NPixel/2] = fftshift(fft2(WF))

# matlab code: WF2 = real(ifft2(fftshift(fftWF2)));
# WF2 = importImages(WF2);
WF2 = np.zeros((NPixel, NPixel))
WF2[:, :] = np.abs(ifft2(fftshift(fftWF2)))

WF2 = import_images1(WF2)











# PCA
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

        