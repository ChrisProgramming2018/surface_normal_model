

import cv2
import numpy as np


def create_normal(d_im, distance=1):
    d_im = d_im.astype("float64")
    data_type = "float64"
    normals = np.zeros((256, 256, 3), dtype=data_type)
    h,w = d_im.shape
    for i in range(distance, w-distance):
        for j in range(distance, h-distance):
            t = (d_im[j, i + distance] - d_im[j, i - distance] ) / (2.0 * distance)
            f = (d_im[j+distance, i] - d_im[j-distance, i] ) / (2.0 * distance)
            direction = np.array([-t,-f,1])
            magnitude = np.sqrt(t**2 + f**2 + 1)
            n = direction / magnitude 
            normals[j,i,:] = n
    return normals
