# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cv2



uploaded = files.upload()

original_img = cv2.imread(list(uploaded.keys())[0], 1)

"""##Показать изображение"""

cv2_imshow(original_img)

"""##Перевод в hsv модель"""

hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

lower_red_1 = np.array([0,80, 15])
upper_red_1 = np.array([12,255,255])

mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

lower_red_2 = np.array([150,80, 15])
upper_red_2 = np.array([180,255,255])

mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

res_mask = cv2.bitwise_or(mask1, mask2)

"""Выделение красного цвета"""

res = cv2.bitwise_and(original_img, original_img, mask = res_mask)

"""Выделение синего цвета"""

res2 = cv2.bitwise_xor(original_img, res)

"""## Результат применения преобразования

###Синий
"""

cv2_imshow(res2)

gray_b = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=5, tileGridSize = (10, 10))
dst_r = clahe.apply(gray_r)
dst_b = clahe.apply(gray_b)

dst_r.mean()

dst_b.mean()

"""###Красный"""

cv2_imshow(res)

"""###Площадь

Количество пикселей красного цвета
"""

k_pixels = res_mask[res_mask > 0].shape[0]

k_pixels

original_img.shape

k_pixels /

"""Доля пикселей красного цвета"""

res_mask[res_mask > 0].shape[0] / (1600 * 1600)

"""###Выделение контуров"""

gray_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=5, tileGridSize = (10, 10))
clahe_gray_image = clahe.apply(gray_img)

ret, bin_img = cv2.threshold(clahe_gray_image, 95, 255, cv2.THRESH_BINARY)

cv2_imshow(bin_img)

kernel = np.ones((4, 4), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
disk = np.array([[0,1,1,0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8)
ring = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], np.uint8)

opening = cv2.dilate(cv2.erode(bin_img, ring), ring)

closing = cv2.erode(cv2.dilate(opening, disk), disk)

final = cv2.erode(closing, kernel2)

def fndCntrs(img, background):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(background, contours, -1, (0,255,0), 3)
    cv2_imshow(background)

fndCntrs(final, res)

fndCntrs(final, test)

"""#Для экспериментов"""

kernel = np.ones((10, 10), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
disk = np.ones(np.uint8)
ring = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], np.uint8)

"""## График распределения"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

def show_hsv_graph(img):
    img2 = img.copy()
    img2 = img2[::-1]
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img1)
    fig = plt.figure(figsize=(8,6),dpi=80)
    axis = fig.add_subplot(projection="3d")
    # Настройки цвета пикселей
    pixel_colors = img2.reshape((np.shape(img2)[0]*np.shape(img2)[1],3))
    # Нормализовано
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    # Преобразовать в список
    pixel_colors = norm(pixel_colors).tolist()
    # Отображение трехмерной диаграммы рассеяния
    axis.scatter(h.flatten(),s.flatten(),v.flatten(),facecolors=pixel_colors,marker='.')
    axis.set_xlabel("hue")
    axis.set_ylabel("saturation")
    axis.set_zlabel("value")
    #axis.plot(h, s, v)
    plt.show()

"""## Выделение форм"""

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

"""### Показать чб картинку"""

cv2_imshow(gray_img)

dst = cv2.equalizeHist(gray_img)

cv2_imshow(dst)

clahe = cv2.createCLAHE(clipLimit=5, tileGridSize = (10, 10))
dst2 = clahe.apply(gray_img)
cv2_imshow(dst2)

gray_r = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

dst_R = clahe.apply(gray_r)

cv2_imshow(dst_R)

"""###Бинаризация"""

import numpy as np
import argparse
import cv2
from google.colab.patches import cv2_imshow

from google.colab import files
uploaded = files.upload()

contr = cv2.imread(list(uploaded.keys())[0], 1)

dst2.mean()

ret, contr = cv2.threshold(dst_R, 95, 255, cv2.THRESH_BINARY)

cv2_imshow(contr)

cv2.THRESH_BINARY

kernel = np.ones((4, 4), np.uint8)

disk = np.array([[0,1,1,0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8)

ring = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], np.uint8)

plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

kernel

disk

ring

plus

dil_img = cv2.dilate(cv2.erode(contr, ring), ring)

cv2_imshow(dil_img)

closing = cv2.erode(cv2.dilate(dil_img, disk), disk)

cv2_imshow(closing)

kernel2 = np.ones((5, 5), np.uint8)

test = cv2.erode(closing, kernel2)

cv2_imshow(test)

test.shape

test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

ret, cntr = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(cntr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours

img_contours = np.zeros(test.shape)
img_contours = cv2.drawContours(test, contours, cv2.RETR_TREE, (244,255,0), cv2.CHAIN_APPROX_SIMPLE)

cv2_imshow(img_contours)

def fndCntrs(img, background):
        #convert img to grey
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    cv2.drawContours(background, contours, -1, (0,255,0), 3)
    cv2_imshow(background)

from google.colab import files
uploaded = files.upload()

img = cv2.imread(list(uploaded.keys())[0], 1)

cv2_imshow(img)

img.shape, test.shape

fndCntrs(test, original_img)

test.shape

uploaded2 = files.upload()

img = cv2.imread(list(uploaded2.keys())[0], 1)

img.shape

cv2_imshow(img)

rs = cv2.bitwise_and(img, test)

test.shape

rs.mean()

rs = cv2.cvtColor(rs, cv2.COLOR_GRAY2BGR)

cv2_imshow(rs)

_, contrast_img = cv2.threshold(gray_img,30,255,cv2.THRESH_BINARY)

cv2_imshow(contrast_img)

gray_red = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

q, red_countour = cv2.threshold(gray_red, 30, 255, cv2.THRESH_BINARY)

cv2_imshow(red_countour)

!pip install scikit-image

from scipy.ndimage import morphology
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import morphology, color, feature, measure, transform, exposure, segmentation, filters, util, draw
import scipy as sp 
import scipy.ndimage as ndi

norm = exposure.equalize_adapthist(original_img)
rey = color.rgb2grey(norm)
block = 71
binary = filters.threshold_local(rey, block_size=block, method='gaussian', mode='reflect')
# binary = ndi.binary_fill_holes(binary) 
# binary = morphology.remove_small_objects(binary, min_size=200) 
# binary = morphology.binary_closing(binary, morphology.disk(1)) 
# binary = ndi.binary_fill_holes(binary) 
# binary = morphology.binary_erosion(binary, morphology.square(5))
# binary = morphology.remove_small_objects(binary, min_size=50)

print(np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1]))

cv2_imshow(norm)

test = morphology.binary_dilation(red_countour)

cv2_imshow(test)

kernel = np.ones((3, 2), np.uint8)
dil = cv2.dilate(red_countour, kernel)

cv2_imshow(dil)

kernel2 = np.ones((4,4), np.uint8)

er = cv2.erode(contrast_img, kernel2)

cv2_imshow(er)

kernel3 = np.ones((1,1), np.uint8)

er2 = cv2.erode(er, kernel3)

dil = cv2.dilate(er2, kernel3)

cv2_imshow(dil)

cv2_imshow(contrast_img)

cv2.gradient

print()