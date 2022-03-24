import cv2
import numpy as np


def get_area(epit_img):
    # scale /= 1000
    original_img = np.array(epit_img.convert('RGB'))
    hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    lower_red_1 = np.array([0, 80, 15])
    upper_red_1 = np.array([12, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([150, 80, 15])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    res_mask = cv2.bitwise_or(mask1, mask2)
    k_pixels = res_mask[res_mask > 0].shape[0]

    return k_pixels


def get_contours(epit_img):
    original_img = np.array(epit_img.convert('RGB'))
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(10, 10))
    clahe_gray_image = clahe.apply(gray_img)
    # gaus_clahe_gray_image = gaussian_filter(clahe_gray_image, sigma=5)
    gaus_clahe_gray_image = cv2.GaussianBlur(clahe_gray_image, (5, 5), 0)
    ret, contr = cv2.threshold(gaus_clahe_gray_image, 95, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(contr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (0, 255, 0), 3)
    return original_img


def mean_shift(epit_img):
    original_img = np.array(epit_img.convert('RGB'))
    spatialRadius = 35
    colorRadius = 60
    pyramidLevels = 3
    res = cv2.pyrMeanShiftFiltering(original_img, spatialRadius, colorRadius, pyramidLevels)
    return res


def area_artefact(artefact_img):
    artefact_img = np.array(artefact_img.convert('RGB'))
    gray = cv2.cvtColor(artefact_img, cv2.COLOR_BGR2GRAY)
    bin = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    return bin[bin > 0].shape[0]

