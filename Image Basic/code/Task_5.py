# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:45:32 2020

@author: Fengdan Cuis

Task - 5 Implement your own 3x3 Sobel filter
"""

import cv2
from Global_lib import display_images, sobel_filter

# load image and convert to gray image

img_gray = cv2.cvtColor(cv2.imread('../face_01_u6589143.jpg'), cv2.COLOR_BGR2GRAY) 

# using my implemented function

my_h = sobel_filter(img_gray, 'h')
my_v = sobel_filter(img_gray, 'v')
my_both = sobel_filter(img_gray)
display_images([my_h, my_v, my_both], ["algong x", "along y", "final result"])


# using in-built function

h = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
v = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
 
abs_h = cv2.convertScaleAbs(h)  
abs_v = cv2.convertScaleAbs(v)

both = cv2.addWeighted(abs_h, 1, abs_v, 1, 0)

display_images([abs_h, abs_v, both], ["inbuilt Sobel algong x", "inbuilt Sobel along y", "inbuilt Sobel final result"])