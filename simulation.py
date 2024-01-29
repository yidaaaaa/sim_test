import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import fft2, ifft2, fftshift
from scipy import signal
import imageio
from sim_otf_provider import sim_otf_provider, otf2psf

size_num = 50
pca_x = np.zeros((3, size_num))
pca_y = np.zeros((3, size_num))
cor_x = np.zeros((3, size_num))
cor_y = np.zeros((3, size_num))

# Parameters
N = 9 # Image number
NA = 0.6 # NA
Mag = 40 # Magnification
Pixelsize = 6.5 / Mag # Pixelszie
wavelength = 561 # Laser wavelength
lambda_ = 0.607 # Fluorescent wavelength
sub_optimization = 0

# load image


Iraw = np.zeros((1024, 1024, N, size_num))  # 增加一个维度来存储不同噪声水平的图像

for error_num in range(size_num):
    for j in range(N):
        # 加载图像
        img = imageio.imread('RawImage/raw.tif', pilmode='F', index=j)
        
        # 添加高斯白噪声
        noise_level = error_num  # 这里可以根据需要调整噪声水平的计算方式
        noise = np.random.normal(0, noise_level, img.shape)
        Iraw[:, :, j, error_num] = img + noise


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


#Preprocessing and widefield image reconstruction
# param.nrDirs=3;                                                             % Number of angles
# param.nrPhases=3;                                                           % Number of phase shifting steps  
# N = param.nrDirs*param.nrPhases;                                            % Number of total phases
# Temp = importImages(Iraw);                                                  % Edge attenuation
# IIraw = deconvlucy(Temp,PSF,5);                                             % Lucy deconvolution

# for I = 1:N
#     IIrawFFT(:,:,I) = FFT2D(IIraw(:,:,I),false);
# end

# WF = zeros(NPixel,NPixel);                                                  % Widefield image sum                      
# [x,y] = meshgrid(1:NPixel,1:NPixel);                                        % Image coordinate
# Tdeconv = zeros(NPixel,NPixel);                                             % Deconvoluted imgae sum
# WFdeconv = zeros(NPixel,NPixel,param.nrDirs);              	                % Deconvoluted imgae
# WFdeconvFFT = zeros(NPixel,NPixel,param.nrDirs);                            % Deconvoluted imgae spectrum               

# for i = 1:param.nrDirs
#     for j =1:param.nrPhases
#         Tdeconv = Tdeconv+IIraw(:,:,(i-1)*param.nrDirs+j);
#         WF(:,:)=WF(:,:)+Iraw(:,:,(i-1)*param.nrDirs+j);
#     end
#     WFdeconv(:,:,i) = Tdeconv/param.nrPhases;
#     WFdeconvFFT(:,:,i) = FFT2D(WFdeconv(:,:,i),false);
# end
# WF = WF/N;
# WF = importImages(WF);   
# figure(),imshow(WF,[]),colormap hot;

param['nrDirs'] = 3
param['nrPhases'] = 3
N = param['nrDirs'] * param['nrPhases']
IIraw = np.zeros((NPixel, NPixel, N, size_num))

for i in range(param['nrDirs']):
    for j in range(param['nrPhases']):
        IIraw[:, :, (i - 1) * param['nrDirs'] + j, :] = Iraw[:, :, (i - 1) * param['nrDirs'] + j, :] / param['nrPhases']

WF = np.zeros((NPixel, NPixel, size_num))
Tdeconv = np.zeros((NPixel, NPixel, size_num))
WFdeconv = np.zeros((NPixel, NPixel, param['nrDirs'], size_num))
WFdeconvFFT = np.zeros((NPixel, NPixel, param['nrDirs'], size_num))

for error_num in range(size_num):
    for i in range(param['nrDirs']):
        for j in range(param['nrPhases']):
            Tdeconv[:, :, error_num] = Tdeconv[:, :, error_num] + IIraw[:, :, (i - 1) * param['nrDirs'] + j, error_num]
            WF[:, :, error_num] = WF[:, :, error_num] + Iraw[:, :, (i - 1) * param['nrDirs'] + j, error_num]
        WFdeconv[:, :, i, error_num] = Tdeconv[:, :, error_num] / param['nrPhases']
        WFdeconvFFT[:, :, i, error_num] = fft2(WFdeconv[:, :, i, error_num])

WF = WF / N
WF = np.abs(fft2(WF))




