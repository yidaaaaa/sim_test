import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import fft2, ifft2, fftshift
from scipy import signal
import imageio
from sim_otf_provider import sim_otf_provider, otf2psf
from importimages import import_images
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

Iraw = np.zeros((frames[0].shape[0], frames[0].shape[1], len(frames), size_num))  # 增加一个维度来存储不同噪声水平的图像

for error_num in range(size_num):
    for i, frame in enumerate(frames):
        # 添加高斯白噪声
        noise_level = error_num  # 噪声水平
        noise = np.random.normal(0, noise_level, frame.shape)
        Iraw[:, :, i, error_num] = frame + noise

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
IIraw = import_images(Iraw)
IIraw_deconv = np.zeros_like(IIraw)
for error_num in range(size_num):
    for i in range(N):
        IIraw_deconv[:, :, i, error_num] = richardson_lucy(IIraw[:, :, i, error_num], PSF, num_iter=10)

IIrawFFT = np.zeros_like(IIraw_deconv, dtype=np.complex128)
for error_num in range(size_num):
    for i in range(N):
        IIrawFFT[:, :, i, error_num] = fft2(IIraw_deconv[:, :, i, error_num])

# for i in range(param['nrDirs']):
#     for j in range(param['nrPhases']):
#         IIraw[:, :, (i - 1) * param['nrDirs'] + j, :] = Iraw[:, :, (i - 1) * param['nrDirs'] + j, :] / param['nrPhases']

WF = np.zeros((NPixel, NPixel, size_num))
Tdeconv = np.zeros((NPixel, NPixel, size_num))
WFdeconv = np.zeros((NPixel, NPixel, param['nrDirs'], size_num))
WFdeconvFFT = np.zeros((NPixel, NPixel, param['nrDirs'], size_num))

for error_num in range(size_num):
    for i in range(param['nrDirs']):
        for j in range(param['nrPhases']):
            Tdeconv[:, :, error_num] = Tdeconv[:, :, error_num] + IIraw[:, :, i * param['nrDirs'] + j+1, error_num]
            WF[:, :, error_num] = WF[:, :, error_num] + Iraw[:, :, i * param['nrDirs'] + j+1, error_num]
        WFdeconv[:, :, i, error_num] = Tdeconv[:, :, error_num] / param['nrPhases']
        WFdeconvFFT[:, :, i, error_num] = fft2(WFdeconv[:, :, i, error_num])

# WF = WF / N
# WF = np.abs(fft2(WF))
for error_num in range(size_num):
    WF[:, :, error_num] = np.abs(fft2(WF[:, :, error_num])) / N

# matlab code: figure(),imshow(WF,[]),colormap hot;

plt.figure()
plt.imshow(WF[:, :, 0], cmap='hot')
plt.show()
    

fftWF2 = np.zeros((2*NPixel, 2*NPixel, size_num))
for error_num in range(size_num):
    fftWF2[NPixel/2 +1:3*NPixel/2, NPixel/2 +1:3*NPixel/2, error_num] = fftshift(fft2(WF[:, :, error_num]))

# matlab code: WF2 = real(ifft2(fftshift(fftWF2)));
# WF2 = importImages(WF2);
WF2 = np.zeros((NPixel, NPixel, size_num))
for error_num in range(size_num):
    WF2[:, :, error_num] = np.abs(ifft2(fftshift(fftWF2[:, :, error_num])))

WF2 = import_images(WF2)



