import cv2
import numpy as np

m = 3
sigma = 1.0
v1 = cv2.getGaussianKernel(m, sigma)
print(v1)
v2 = np.transpose(v1)
print(v2)
w = np.matmul(v1, v2)
print(w)
