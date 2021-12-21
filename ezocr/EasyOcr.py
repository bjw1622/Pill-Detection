import easyocr
import cv2
from matplotlib import pyplot as plt
from pylab import rcParams
import numpy as np

def run(img):
    # find the white rectangle
    th = img.copy()
    th[th<200] = 0

    bbox = np.where(th>0)
    y0 = bbox[0].min()
    y1 = bbox[0].max()
    x0 = bbox[1].min()
    x1 = bbox[1].max()

    # crop the region of interest (ROI)
    img = img[y0:y1, x0:x1]

    # histogram equalization
    equ = cv2.equalizeHist(img)
    # Gaussian blur
    blur = cv2.GaussianBlur(equ, (5, 5), 1)


    # manual thresholding
    th2 = 90 # this threshold might vary!
    equ[equ>=th2] = 255
    equ[equ<th2]  = 0



    # First time for blur image
    rcParams['figure.figsize'] = 8, 16
    reader = easyocr.Reader(['en'])

    output = reader.readtext(blur)

    for i in range(len(output)):
        print(output[i][-2])

    # Second time for equ image
    rcParams['figure.figsize'] = 8, 16
    reader = easyocr.Reader(['en'])

    output = reader.readtext(equ)

    for i in range(len(output)):
        print(output[i][-2])

    plt.subplot(1,2,1) #for blur image
    plt.imshow(blur, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.subplot(1,2,2) #for blur image
    plt.imshow(equ, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()