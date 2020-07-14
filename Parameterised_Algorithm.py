import cv2
import numpy as np
import metrics
import matplotlib.pyplot as plt
from metrics import *
import csv
import logging


# Convert the ground truth into a mask
GT = cv2.imread("Data/Florida_GT.png")
GT = np.array(GT)

# Select one channel
GT = GT[:, :, -1]

# Generate a binary mask
GT[GT > 0] = 1
plt.imshow(GT,  cmap="Blues")
plt.show()

# Import/Extract MSI bands
MSI = np.load(f"Data/Florida.npy")
blue = MSI[-1][:, :, 1]
green = MSI[-1][:, :, 2]
red = MSI[-1][:, :, 3]
NIR = MSI[-1][:, :, 7]
SWIR1 = MSI[-1][:, :, 10]
SWIR2 = MSI[-1][:, :, 11]

# Generate a water mask with a given parameter
# Water Index
l = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
miou = []
pa = []

for val in l:
    i = val   # NDBI
    j = 0.75    # MDWI.1
    k = 1   # MDWI.2

    WI = ((i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
          (j * ((green - SWIR2) / (green + SWIR2))) +
          (k * ((green - NIR) / (green + NIR))))
    WI[WI > 0] = 0
    WI[WI < 0] = 1

    # plt.imshow(WI, cmap="Blues")
    # plt.show()

    # Evaluate segmentation output
    miou.append(mean_IU(eval_segm=WI, gt_segm=GT))
    pa.append(pixel_accuracy(eval_segm=WI, gt_segm=GT))

plt.plot(l, miou)
plt.plot(l, pa)
plt.xlabel("Scalar")
plt.ylabel("MIOU")
plt.title(f"Scalar applied to MNDWI, with NDBI set to 0.75")
plt.show()










