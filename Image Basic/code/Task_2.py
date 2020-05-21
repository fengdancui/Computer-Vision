# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:44:01 2020

@author: Fengdan Cui

Task - 2 Basic Coding Practice 
"""

import numpy as np
import imageio
import random
from Global_lib import display_images
#import cv2
        
file = 'Lenna.png'
file_col = 'Apples.jpg'

# load image 

img = imageio.imread(file)
img_gray = np.dot(img[:, :, :3], [0.299, 0.587, 0.114])

# if using cv2
# cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# map the image to its negative image

img_neg = np.uint8(255 - img_gray)
display_images([img, img_gray, img_neg], ["original image", "grayscale image", "negative image"])

display_images([img_gray, np.flipud(img_gray)], ["original grayscale image", "after flipping vertically"])

# load a colour image, swap the red and blue colour channels of the input

img_col = imageio.imread(file_col)
img_col_change = img_col[:, :, (2, 1, 0)]

display_images([img_col, img_col_change], ["original image", "after changing channel"])

# if using cv2
# cv2.cvtColor(img_col, cv2.COLOR_RGB2BGR)

# average the input image with its vertically flipped image 

img_col_ver = np.flipud(img_col)
img_col_avg= np.uint8(np.average([img_col, img_col_ver], axis = 0))
display_images([img_col, img_col_ver, img_col_avg], ["original image", "after flipping vertically", "after averaging"])

# add a random value in the grayscale image and then clip the new image
img_gray_thr = img_gray.copy()
rows, cols = img_gray_thr.shape

for c in range(cols):
    for r in range(rows):
        rand = random.randint(0, 255)
        img_gray_thr[r,c] += rand
        
img_gray_thr = np.uint8(np.clip(img_gray_thr, 0, 255))   
            
display_images([img_gray, img_gray_thr], ["original image", "after adding random"])


        