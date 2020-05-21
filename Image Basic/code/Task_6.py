# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:19:06 2020

@author: Fengdan Cui

task - 6  Image Rotation 
"""
import cv2
from Global_lib import display_images, my_rotation, nearest_inter, linear_inter

# load image and convert to RGB and resize to 512 * 512

img = cv2.cvtColor(cv2.imread('../face_01_u6589143.jpg'), cv2.COLOR_BGR2RGB)
img_rsz = cv2.resize(img, (512, 512))

# display images rotated by -90, -45, -15, 45, and 90

rotated_img_n90 = my_rotation(img_rsz, -90)
rotated_img_n45 = my_rotation(img_rsz, -45)
rotated_img_n15 = my_rotation(img_rsz, -15)
rotated_img_45 = my_rotation(img_rsz, 45)
rotated_img_90 = my_rotation(img_rsz, 90)

display_images([img_rsz, rotated_img_n90, rotated_img_n45, rotated_img_n15, rotated_img_45, rotated_img_90], 
               ["original image", "rotated -90", "rotated -45", "rotated -15", "rotated 45", "rotated 90"])

rotated_img_n90_b = my_rotation(img_rsz, -90, 'b')
rotated_img_n45_b = my_rotation(img_rsz, -45, 'b')
rotated_img_n15_b = my_rotation(img_rsz, -15, 'b')
rotated_img_45_b = my_rotation(img_rsz, 45, 'b')
rotated_img_90_b = my_rotation(img_rsz, 90, 'b')

display_images([img_rsz, rotated_img_n90_b, rotated_img_n45_b, rotated_img_n15_b, rotated_img_45_b, rotated_img_90_b], 
               ["original image", "rotated -90", "rotated -45", "rotated -15", "rotated 45", "rotated 90"])


display_images([rotated_img_45, rotated_img_45_b], ["rotated 45 by forward mapping", "rotated 45 by backward mapping"])

# different interpolation methods for rotation

rotated_img_n45 = my_rotation(img_rsz, -45)
nearest_inter = nearest_inter(img_rsz, rotated_img_n45, -45)
linear_inter = linear_inter(img_rsz, rotated_img_n45, -45)

display_images([rotated_img_n45, nearest_inter, linear_inter], 
               ["original rotated -45", "after nearest-neighbour interpolation", "after bilinear interpolation"])

# different interpolation methods in cv2 for resizing

img_crop_n = cv2.resize(img[200: 300, 400: 500], (256, 256), interpolation = cv2.INTER_NEAREST)
img_crop_l = cv2.resize(img[200: 300, 400: 500], (256, 256), interpolation = cv2.INTER_LINEAR)

display_images([img_crop_n, img_crop_l], ["resize image using nearest-neighbor", "resize image using bilinear interpolation"])