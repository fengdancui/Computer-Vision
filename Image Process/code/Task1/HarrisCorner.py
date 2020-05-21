"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Fengdan Cui (u6589143)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01


'''
# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()


bw = plt.imread('Harris_1.pgm')
bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)
'''

######################################################################
# Task: Compute the Harris Cornerness
# Compute corner response R 

def harris_corneress(image, sigma = 2, k = 0.04):
    
    h, w = image.shape
    
    # R to store cornerness score    
    R = np.zeros((h, w))
    
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = dx.transpose()

    # calculating gradients in x and y direction  
    Ix = conv2(image, dx)
    Iy = conv2(image, dy)
    
    # appLy Gaussian filter
    g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
    Iy2 = conv2(np.power(Iy, 2), g)
    Ix2 = conv2(np.power(Ix, 2), g)
    Ixy = conv2(Ix * Iy, g)
    
    for i in range(h):
        for j in range(w):
            
            # harris matrix
            M = [[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]]
            
            # singular value decomposition
            u, s, v = np.linalg.svd(M)
            
            # omputing lambda values
            [lmda1, lmda2] = s
            lambda_product = lmda1 * lmda2
            lambda_sum = lmda1 + lmda2
            
            # computing corneress scores and store in R
            R[i, j] = lambda_product - k * (lambda_sum**2)
            
    return R

######################################################################


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
# Find points with large corner response: R > threshold
# Take only the points of local maxima of R
    
def non_max_sup(R, thresh = 0.01):
    
    # get the height and width of image for interation later, the shape of R is the same as image
    h, w = R.shape
    
    # considering only those points that are larger than threshold
    thresh *= R.max()
    
    # an Nx2 matrix for storing the final x and y coordinates
    out = list()
    
    # 3 x 3 window, start from 1 to len - 1 for local maxima of each window
    for r in range(1, h - 1):
        for c in range(1, w - 1):  
            
            # If the point is greater than the largest point among the surrounding points, 
            # then it is greater than all the points around
            Rmax = 0
            
            # iterating all surrounding points 3x3
            for i in [r - 1, r + 1]:
                for j in [c - 1, c + 1]:
                    
                    # comparing and replacing max
                    if R[i, j] > Rmax:
                        Rmax = R[i, j]
                        
            # pick out points of local maxima and larger than threshold
            if (R[r, c] > thresh) & (R[r, c] >= Rmax):  
                
                # the index in matrix is the reverse order of the coordinates of image, 
                # so add [c, r] not [r, c] to list
                out.append([c, r])

    return out

######################################################################
    
# using implemented function 

def corner_detection(img_files):
    fig, axs = plt.subplots(figsize = (20, 15), ncols = 2, nrows = 2)
    
    for i, file in enumerate(img_files):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        
        print('looking for corners ......')
        
        # call implemented function for corner detection and then local max caculation
        corners = non_max_sup(harris_corneress(img_gray))
        
        x = 0 if i < 2 else 1
        
        axs[x][i - x * 2].imshow(img)
        axs[x][i - x * 2].set_title('Image ' + str(i + 1))
        for j in range(len(corners)):
            
            # draw empty circles at the location of corners
            axs[x][i - x * 2].add_patch(plt.Circle(corners[j], 2, color = 'r', fill = False))
    
    fig.suptitle('Corner detection using implemented function', fontsize = 18)

# using ibuilt function cv2.cornerHarris
    
def corner_detection_cv(img_files):
    fig, axs = plt.subplots(figsize = (20, 15), ncols = 2, nrows = 2)
    
    for i, file in enumerate(img_files):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        
        # calling inbuilt function for corner detection
        corners_cv = cv2.cornerHarris(img_gray, 3, 3, 0.04)
        
        # only select those features larger than threshold
        img[corners_cv > thresh * corners_cv.max()] = [255, 0, 0]
        
        x = 0 if i < 2 else 1
            
        axs[x][i - x * 2].imshow(img)
        axs[x][i - x * 2].set_title('Image ' + str(i + 1))
    fig.suptitle('Corner detection using inbuilt function', fontsize = 18)

# call function and display images
    
corner_detection(['Harris_1.pgm', 'Harris_1.jpg', 'Harris_3.jpg', 'Harris_4.jpg'])
corner_detection_cv(['Harris_1.pgm', 'Harris_1.jpg', 'Harris_3.jpg', 'Harris_4.jpg'])


# reseach in the factors that affect the performance of Harris
# scale change

img = cv2.cvtColor(cv2.imread('Harris_3.jpg'), cv2.COLOR_BGR2RGB)
img_crop = cv2.resize(img[200: 300, 400: 500], (100 * 50, 100 * 50), interpolation = cv2.INTER_LINEAR)
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
img_crop_gray = np.float32(cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY))

fig, axs = plt.subplots(figsize = (20, 15), ncols = 2)
axs[0].imshow(img)
axs[1].imshow(img_crop)

corners = cv2.cornerHarris(img_gray, 3, 3, 0.04)
img[corners > thresh * corners.max()] = [255, 0, 0]
corners_crop = cv2.cornerHarris(img_crop_gray, 3, 3, 0.04)
img_crop[corners_crop > thresh * corners_crop.max()] = [255, 0, 0]

fig, axs = plt.subplots(figsize = (20, 15), ncols = 2)
axs[0].imshow(img)
axs[0].set_title('original image corners detection')
axs[1].imshow(img_crop)
axs[1].set_title('cropped and enlarged image corners detection')