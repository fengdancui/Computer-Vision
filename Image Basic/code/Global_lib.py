# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:45:16 2020

@author: Fengdan Cui

all necessry functions used for completing tasks
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

# gaussian filter implement for removing noise by caculating convolution 

def my_Gauss_filter(noisy_image, size, sigma):
    
    kernel_x = cv2.getGaussianKernel(size, sigma).reshape(size, 1)
    kernel_y = cv2.getGaussianKernel(size, sigma).reshape(1, size)
    kernel =  kernel_x * kernel_y
    out = caculate_conv(noisy_image, kernel)
    return out


# sobel filter implement for edge detection in different direction by caculating convolution

def sobel_filter(image, direction = 'both'):
    print("processing sobel filter along", direction)
    
    mask_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    mask_h = mask_v.T
    filter_h = np.abs(caculate_conv(image, mask_h)).astype(np.uint8)
    filter_v = np.abs(caculate_conv(image, mask_v)).astype(np.uint8)
    filter_all = (filter_h + filter_v).astype(np.uint8)
    
    if direction == 'h':
        return filter_h
    elif direction == 'v':
        return filter_v
    return filter_all

# caculating convolution by dot and sum 
# add padding for input in order to ensure the size of output is the same as input
                
def caculate_conv(inputs, mask):
    
    out = np.zeros(inputs.shape)
    size = mask.shape[0]
    pad_len = math.floor(size / 2)
    
    # (rows, cols) -> (rows + 2, cols + 2)
    inputs_pad = np.pad(inputs, ((pad_len, pad_len), (pad_len, pad_len)),'constant', constant_values=(0, 0)) 
    h, w = inputs_pad.shape
    
    for i in range(h - size + 1):
        for j in range(w - size + 1):
            out[i, j] = np.sum((inputs_pad[i: i + size, j: j + size]) * mask)
    
    return out

# rotate image according to give angle using different methods
# out: rotetion output of forward mapping
# out_b: rotetion output of backward mapping

def my_rotation(image, angle, mapping = 'f'):     
    print("processing rotating for", angle)

    h, w, c = image.shape
    cx = h / 2
    cy = w / 2
    out = np.zeros(image.shape)
    out_b = np.zeros(image.shape)
    r_matrix = rotation_matrix(angle)
    
    for i in range(h):
        for j in range(w):
            
            # rotation center is the center of the image
            
            xy = np.dot(np.array([i - cx, j - cy]), r_matrix)
            x = int(xy[0] + cx)
            y = int(xy[1] + cy)
            
            # the piexls of the rotated which are out the bound of image can't be shown
             
            if x > -1 and x < h and y > -1 and y < w:
                out[x, y, :] = image[i, j, :]     
               
            xy_b = np.dot(np.array([i - cx, j - cy]), np.linalg.inv(r_matrix))
            x_b = int(xy_b[0] + cx)
            y_b = int(xy_b[1] + cy)
            
            if x_b > -1 and x_b < h and y_b > -1 and y_b < w:
                out_b[i, j, :] = image[x_b, y_b, :] 
           
    
    if mapping == 'b':
        return np.uint8(out_b)
    
    return np.uint8(out)

# 2d rotation matrix 

def rotation_matrix(angle):
    
    angle_pi = angle * math.pi/180.0 
    cos = math.cos(angle_pi)
    sin = math.sin(angle_pi)

    return np.array([[cos, -sin],
                     [sin, cos]])    
    
# nearest-neighbor interpolation implement
# rounds real-valued coordinates calculated by a geometric transformation to their nearest integers
    
def nearest_inter(image_in, image_out, angle):
    print("processing nearest-neighbor interpolation for", angle)
    
    rot_matrix = rotation_matrix(angle)
    h, w, c = image_out.shape
    cx = h / 2
    cy = w / 2
    out = image_out.copy()
    
    for i in range(h):
        for j in range(w):
            if image_out[i, j].all() == 0:
                xy = np.dot(np.array([i - cx, j - cy]), np.linalg.inv(rot_matrix))
                
                # found out the nearest point
                
                x = math.ceil(xy[0] - 0.5 + cx)
                y = math.ceil(xy[1] - 0.5 + cy)
                
                if x > -1 and x < h and y > -1 and y < w:
                    out[i, j, :] = image_in[x, y, :]
                
    return np.uint8(out)

# bilinear interpolation implement
# computed as a hyperbolic distance-weighted function of the four pixels in integer positions (x0,y0), 
# (x1,y0), (x0,y1), and (x1,y1), surrounding the calculated real-valued position (x,y). 

def linear_inter(image_in, image_out, angle):
    print("processing bilinear interpolation for", angle)
    
    rot_matrix = rotation_matrix(angle)
    h, w, c = image_out.shape
    cx = h / 2
    cy = w / 2
    out = image_out.copy()
    
    for k in range(c):
        for i in range(h):
            for j in range(w):
                if image_out[i, j].all() == 0:
                    xy = np.dot(np.array([i - cx, j - cy]), np.linalg.inv(rot_matrix))
                    x = xy[0] + cx
                    y = xy[1] + cy
                    xf = math.floor(x)
                    yf = math.floor(y)
                    
                    if xf > -1 and xf < h - 1 and yf > -1 and yf < w - 1:
                        
                        # four pixels in integer positions (x0,y0), (x1,y0), (x0,y1), and (x1,y1)
                        point1 = (xf, yf)
                        point2 = (xf, yf + 1)
                        point3 = (xf + 1, yf)
                        point4 = (xf + 1, yf + 1)
                        
                        # horizontal interpolation
                        
                        fr1 = (point2[1] - y) * image_in[point1[0], point1[1], k] + (y - point1[1]) * image_in[point2[0], point2[1], k]
                        fr2 = (point4[1] - y) * image_in[point3[0], point3[1], k] + (y - point1[1]) * image_in[point4[0], point4[1], k]
                              
                        # vertical interpolation
                        
                        out[i, j, k] = (point3[0] - x) * fr1 + (x - point1[0]) * fr2
    return np.uint8(out)

# display all images in a list 

def display_images(images, titles):
    fig, axs = plt.subplots(figsize=(6 * len(images), 5), ncols = len(images))
    if len(images) > 1:
        for i, image in enumerate(images):
            if image.ndim == 2:
                axs[i].imshow(images[i], cmap = 'gray')
            else:
                axs[i].imshow(images[i])
            axs[i].set_title(titles[i])
    else:
        if images[0].ndim == 2:
            axs.imshow(images[0], cmap = 'gray')
        else:
            axs.imshow(images[0])
        axs.set_title(titles[0])
        
        
# another way for caculating rotation matrix 

"""
m = np.array([[cos, -sin, 0],
              [sin, cos, 0],
              [0, 0, 1]])
    
c = np.array([[1, 0, -cx],
              [0, 1, -cy],
              [0, 0, 1]])
xy = np.dot(np.dot(np.linalg.inv(m), c), np.array([i, j, 1]).T)
    
"""
    