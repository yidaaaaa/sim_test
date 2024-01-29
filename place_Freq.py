import numpy as np

def place_freq(in_matrix):
    h, w = in_matrix.shape
    out_matrix = np.zeros((2 * h, 2 * w), dtype=in_matrix.dtype)
    out_matrix[h // 2: h // 2 + h, w // 2: w // 2 + w] = in_matrix
    return out_matrix
