# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:28:47 2020

@author: Fengdan Cui

Task - 4 Image Denoising via a Gaussian Filter 
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Global_lib import display_images, my_Gauss_filter

# load image and resize and convert to gray image

img = cv2.cvtColor(cv2.imread('../face_01_u6589143.jpg'), cv2.COLOR_RGB2BGR)
img_crop = cv2.resize(img[0: 600, 250: 850], (256, 256))
img_crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
display_images([img, img_crop, img_crop_gray], ["original image", "cropped image", "cropped grayscale image"])

# save the new image

cv2.imwrite("face_01_crop.jpg", img_crop_gray)

# add Gaussian noise to this new 256x256 image 

gaussian = np.random.normal(0, 15, img_crop_gray.shape)
img_noise = img_crop_gray + gaussian
# cv2.normalize(img_noise, img_noise, 0, 255, cv2.NORM_MINMAX)
display_images([img_crop_gray, img_noise], ["original grayscale image", "after adding gaussian noise"])

# display the two histograms side by side, one before adding the noise and one after adding the noise 

fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols = 2)
ax1.hist(img_crop_gray.ravel(), 256, [0,256])
ax1.set_title("histogram before adding noise")
ax2.hist(img_noise.ravel(), 256, [0,256])
ax2.set_title("histogram after adding noise")

# apply my gaussian filter to the above noisy image

sig_1 = my_Gauss_filter(img_noise, 5, 1)
sig_3 = my_Gauss_filter(img_noise, 5, 3)
sig_5 = my_Gauss_filter(img_noise, 5, 5)

display_images([sig_1, sig_3, sig_5], 
              ["sigma = 1", "sigma = 3", "sigma = 5"])

sig_7 = my_Gauss_filter(img_noise, 5, 7)
sig_9 = my_Gauss_filter(img_noise, 5, 9)
sig_11 = my_Gauss_filter(img_noise, 5, 11)

display_images([sig_7, sig_9, sig_11], 
              ["sigma = 7", "sigma = 9", "sigma = 11"])

sig_13 = my_Gauss_filter(img_noise, 5, 13)
sig_15 = my_Gauss_filter(img_noise, 5, 15)
sig_17 = my_Gauss_filter(img_noise, 5, 17)

display_images([sig_13, sig_15, sig_17], 
              ["sigma = 13", "sigma = 15", "sigma = 17"])

# using inbuilt function 

sig_1_cv = cv2.GaussianBlur(img_noise, (5, 5), 1)
sig_5_cv = cv2.GaussianBlur(img_noise, (5, 5), 5)
sig_7_cv = cv2.GaussianBlur(img_noise, (5, 5), 7)
display_images([sig_1, sig_5, sig_7], 
              ["sigma = 1", "sigma = 5", "sigma = 10"])
display_images([sig_1_cv, sig_5_cv, sig_7_cv], 
              ["cv sigma = 1", "cv sigma = 5", "cv sigma = 10"])


    
    
    

"""
def main(): 
   
   
    
if __name__ == "__main__": 
    main() 
""" 