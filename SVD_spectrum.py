import numpy as np
from scipy.linalg import svd
from scipy import unwrap
from numpy.fft import ifft2, ifftshift

def svd_spectrum(ROI, Filter_size):
    SIZE = 2 * Filter_size + 1
    Space_ROI = ifft2(ifftshift(ROI))
    Phase_ROI = np.exp(1j * np.angle(Space_ROI))

    U, S, V = svd(Phase_ROI, full_matrices=True)
    S_matrix = np.zeros((SIZE, SIZE))
    S_matrix[0, 0] = S[0]

    Phase_ROI = U @ np.diag(S_matrix) @ V

    UNV1 = unwrap(np.angle(V[:, 0]))
    UNV_T_fit = UNV1[1:SIZE]

    FIT = np.polyfit(range(1, SIZE), UNV_T_fit, 1)
    dx = -FIT[0] * SIZE / (2 * np.pi)
    NewUNV1 = np.polyval(FIT, range(1, SIZE + 1))

    NEWV1 = np.exp(1j * NewUNV1)
    V[:, 0] = NEWV1

    UNU1 = unwrap(np.angle(U[:, 0]))
    UNN_T_fit = UNU1[1:SIZE]

    FIT = np.polyfit(range(1, SIZE), UNN_T_fit, 1)
    dy = FIT[0] * SIZE / (2 * np.pi)
    NewUN1 = np.polyval(FIT, range(1, SIZE + 1))

    NEWU1 = np.exp(1j * NewUN1)
    U[:, 0] = NEWU1

    NEW_Phase_ROI = U @ np.diag(S_matrix) @ V

    return NEW_Phase_ROI, dx, dy
