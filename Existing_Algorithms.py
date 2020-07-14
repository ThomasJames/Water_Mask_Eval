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
plt.imshow(GT, cmap="Blues")
plt.show()

# Import/Extract MSI bands
MSI = np.load(f"Data/Florida.npy")
blue = MSI[-1][:, :, 1]
green = MSI[-1][:, :, 2]
red = MSI[-1][:, :, 3]
NIR = MSI[-1][:, :, 7]
SWIR1 = MSI[-1][:, :, 10]
SWIR2 = MSI[-1][:, :, 11]


# Normalised built-up difference index
NBDI = (SWIR2 - NIR)/(SWIR2 + NIR)
NBDI[NBDI > 0] = 0
NBDI[NBDI < 0] = 1
NBDI_miou = mean_IU(eval_segm=NBDI, gt_segm=GT)
NBDI_pa = pixel_accuracy(eval_segm=NBDI, gt_segm=GT)
NBDI_ma = mean_accuracy(eval_segm=NBDI, gt_segm=GT)
NBDI_reference = "McFeeters (1996)"

# Normalised Water difference index
NWDI = (green - NIR)/(green + NIR)
NWDI[NWDI > 0] = 0
NWDI[NWDI < 0] = 1
NWDI_miou = mean_IU(eval_segm=NWDI, gt_segm=GT)
NWDI_pa = pixel_accuracy(eval_segm=NWDI, gt_segm=GT)
NWDI_ma = mean_accuracy(eval_segm=NWDI, gt_segm=GT)
NWDI_reference = "Zha et al., (2003)"

# Modified Normalised Water difference index
MNDWI = (green - SWIR2)/(green + SWIR2)
MNDWI[MNDWI > 0] = 0
MNDWI[MNDWI < 0] = 1
MNDWI_miou = mean_IU(eval_segm=MNDWI, gt_segm=GT)
MNDWI_pa = pixel_accuracy(eval_segm=MNDWI, gt_segm=GT)
MNDWI_ma = mean_accuracy(eval_segm=MNDWI, gt_segm=GT)
MNDWI_reference = "Xu (2006)"

I = ((green - NIR)/(green + NIR)) + ((blue - NIR)/(blue + NIR))
I[I > 0] = 0
I[I < 0] = 1
I_miou = mean_IU(eval_segm=I, gt_segm=GT)
I_pa = pixel_accuracy(eval_segm=I, gt_segm=GT)
I_ma = mean_accuracy(eval_segm=I, gt_segm=GT)
I_reference = "Mishra & Prasad (2015)"

# Proposed index
PI = ((green - SWIR2)/(green + SWIR2)) + ((blue - NIR)/(blue + NIR))
PI[PI > 0] = 0
PI[PI < 0] = 1
PI_miou = mean_IU(eval_segm=PI, gt_segm=GT)
PI_pa = pixel_accuracy(eval_segm=PI, gt_segm=GT)
PI_ma = mean_accuracy(eval_segm=PI, gt_segm=GT)
PI_reference = "Jain et al., 2020"

# Organise data for plots
eval = [NBDI_reference, NWDI_reference, MNDWI_reference, I_reference, PI_reference]
eval_miou = [NBDI_miou, NWDI_miou, MNDWI_miou, I_miou, PI_miou]
eval_pa = [NBDI_pa, NWDI_pa, MNDWI_pa, I_pa, PI_pa]
eval_ma = [NBDI_ma, NWDI_ma, MNDWI_ma, I_ma, PI_ma]

# Metric vs Algorithm
plt.bar(eval, eval_miou)
plt.xlabel("Algorithm")
plt.ylabel("miou")
plt.show()

plt.bar(eval, eval_pa)
plt.xlabel("Algorithm")
plt.ylabel("Pixel Accuracy")
plt.show()

plt.bar(eval, eval_ma)
plt.xlabel("Algorithm")
plt.ylabel("Mean Accuracy")
plt.show()



