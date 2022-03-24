import argparse
from pathlib import Path

import numpy as np
import cv2

from configs.config import UPPER_RED_COLOR_BORDERS, LOWER_RED_COLOR_BORDERS, STORAGE_PATH


def define_color_in_image(img, id_, color="red"):
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(str(STORAGE_PATH / f"origin_{id_}.jpg"), img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv_img, *map(
        np.array, (LOWER_RED_COLOR_BORDERS[0], UPPER_RED_COLOR_BORDERS[0])
        )
    )

    mask2 = cv2.inRange(
        hsv_img, *map(
        np.array, (LOWER_RED_COLOR_BORDERS[1], UPPER_RED_COLOR_BORDERS[1])
        )
    )

    res_mask = cv2.bitwise_or(mask1, mask2)

    result_red = cv2.bitwise_and(img, img, mask=res_mask)
    result_blue = cv2.bitwise_xor(img, result_red)

    # gray_blue = cv2.cvtColor(result_blue, cv2.COLOR_BGR2GRAY)
    # gray_red = cv2.cvtColor(result_red, cv2.COLOR_BGR2GRAY)

    # clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(10, 10))
    # dst_r = clahe.apply(gray_red)
    # dst_b = clahe.apply(gray_blue)

    red_pixels = res_mask[res_mask > 0].shape[0]
    ratio_red_to_square = red_pixels / (img.shape[0] * img.shape[1])

    cv2.imwrite(f"{STORAGE_PATH}/red_picture_{id_}.jpg", result_red)
    cv2.imwrite(f"{STORAGE_PATH}/blue_picture_{id_}.jpg", result_blue)
    with open(STORAGE_PATH / f"{id_}.txt", "w") as info_file:
        info_file.write(str(ratio_red_to_square))

    return result_red, result_blue, ratio_red_to_square


