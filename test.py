import numpy as np
import cv2
from sklearn.decomposition import PCA
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import IncrementalPCA

file_path = 'C:/Users/yida/OneDrive/code_onedrive/SIM清华/Supplementary code and data/Test/RawImage/Ours/561_1.tif'
tif = Image.open(file_path)

images = []
try:
    while True:
        images.append(np.array(tif))
        tif.seek(tif.tell() + 1)
except EOFError:
    pass

processed_images = [cv2.resize(img, (1024, 1024)) for img in images]

# plt.imshow(processed_images[0], cmap='gray')
# plt.show()

data = np.array(images).reshape(len(images), -1)

mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0)
data_normalized = (data - mean) / std_dev

n_components = 2  
ipca = IncrementalPCA(n_components=n_components, batch_size=10) 
ipca.fit(data_normalized)
transformed_data = ipca.transform(data_normalized)

print(ipca.components_)

n_components = ipca.components_.shape[0]
for i in range(n_components):
    plt.imshow(ipca.components_[i].reshape(1024, 1024), cmap='gray')  
    plt.title(f"Principal Component {i+1}")
    plt.colorbar()
    plt.show()