import numpy as np
import imageio
from scipy import fftpack
from skimage.restoration import richardson_lucy
from scipy.ndimage import fourier_gaussian
import matplotlib.pyplot as plt

# Set internal parameters
size_num = 50
pca_x = np.zeros((3, size_num))
pca_y = np.zeros((3, size_num))
cor_x = np.zeros((3, size_num))
cor_y = np.zeros((3, size_num))

N = 9  # Image number
NA = 0.6  # NA
Mag = 40  # Magnification
Pixelsize = 6.5 / Mag  # Pixelsize
wavelength = 561  # Laser wavelength
lambda_ = 0.607  # Fluorescent wavelength
sub_optimization = 0
