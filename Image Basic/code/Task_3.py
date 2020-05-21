# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:59:30 2020

@author: Fengdan Cui

Task - 3 Basic Image I/O 
"""

import matplotlib.pyplot as plt
import cv2

from Global_lib import display_images

# load image, convert to RGB, resize to 768 x 512

img_bgr = cv2.imread('../face_01_u6589143.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rsz = cv2.resize(img_rgb, (768, 512))

display_images([img_rgb, img_rsz], ["original RGB image with shape " + str(img_rgb.shape), "resized image with shape " + str(img_rsz.shape)])

# convert the colour image into three grayscale channels

blue, green, red = cv2.split(img_rsz)

display_images([blue, green, red], ["blue channel", "green channel", "red channel"])

# compute the histograms for each of the grayscale images

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 5), ncols = 3)
ax1.hist(blue.ravel(), 256, [0,256])
ax1.set_title("histogram of blue channel")
ax2.hist(green.ravel(), 256, [0,256])
ax2.set_title("histogram of green channel")
ax3.hist(red.ravel(), 256, [0,256])
ax3.set_title("histogram of red channel")

# apply histogram equalisation to the resized image and its three grayscale channels

img_ycr_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2YCrCb)
channels = cv2.split(img_ycr_rsz)
channels[0] = cv2.equalizeHist(channels[0])

img_mer = cv2.merge(channels)
equ = cv2.cvtColor(img_mer, cv2.COLOR_YCR_CB2BGR)

equ_blue = cv2.equalizeHist(blue)
equ_green = cv2.equalizeHist(green)
equ_red = cv2.equalizeHist(red)

display_images([equ, equ_blue, equ_green, equ_red], ["HE of resized image", "HE of blue channel", "HE of green channel", "HE of red channel"])

fig, (ax4, ax5, ax6, ax7) = plt.subplots(figsize=(24, 5), ncols = 4)

# histograms of the images after applying HE

color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([equ], [i], None, [256], [0, 256])
    ax4.plot(histr, color = col)
    ax4.set_xlim([0,256])

ax4.set_title("histogram of HE resized image")
ax5.hist(equ_blue.ravel(), 256, [0,256])
ax5.set_title("histogram of HE blue channel")
ax6.hist(equ_green.ravel(), 256, [0,256])
ax6.set_title("histogram of HE green channel")
ax7.hist(equ_red.ravel(), 256, [0,256])
ax7.set_title("histogram of HE red channel")