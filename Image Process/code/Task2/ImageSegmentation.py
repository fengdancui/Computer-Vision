# -*- coding: utf-8 -*-
"""
CLAB Task-2: K-Means Clustering and Color Image Segmentation
Your name (Your uniID): Fengdan Cui (u6589143)
"""

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

# caculate the distance between two points / samples
def dist_eclu(a, b):
    return np.sqrt(sum(np.power(a - b, 2)))


# Implement K-means function
    
def my_kmeans(samples, k, init = 'random', max_iter = 10):
    
    # number of training: r = samples.shape[0]
    # number of features: c = samples.shape[1]
    r, c = samples.shape
    
    # centers initialization by different methods
    if init == 'random':
        
        rands = random.sample(range(0, r), k)   # generate different randoms
        centers = samples[rands, :]             # samples.shape[1] x k
        
    if init == 'plus':
        centers = kmeans_plus_init(samples, k)
    
    # store clusters {0: [points], 1: [points], ..., k: [points]}
    clusters = {i: [] for i in range(k)}
    n = 0
    
    while n < max_iter:
        labels = []
        
        # generate clusters
        for i, s in enumerate(samples):
            distance = []
            for ce in centers:
                distance.append(dist_eclu(s, ce))
            
            # find out the minimal distance and return the corresponding index - one of the centers
            min_i = np.argmin(distance)
            
            # store the sample in dictionary under the key (one center)
            clusters[min_i].append(s)
            
            # give a label for a smaple, 0, 1, 2, ..., k
            labels.append(min_i)

        # recalculate the central value
        centers = np.zeros((k, c))
        for i, value in enumerate(clusters.values()): 
            mean = np.mean(value, axis = 0)
            centers[i] = mean
        
        print("iterating " + str(n + 1) + " time(s) ......")  
    
        n += 1
    
    return np.array(labels), centers


# encode the pixel into 5-D vector and return an array store all vectors, 2D -> (h x w) x 5
def image_encode(image, coordinates = True):
    
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    h, w, c = img_lab.shape
    L, A, B = cv2.split(img_lab)
    
    pixel_vector = np.vstack((L.flatten(), A.flatten(), B.flatten()))
    
    if coordinates:
        x, y = np.indices((h, w))
        pixel_vector = np.vstack((pixel_vector, x.flatten(), y.flatten())).T
    else:
        pixel_vector = pixel_vector.T
        
    return np.array(pixel_vector)


# kmeans++ for init centers
def kmeans_plus_init(samples, k):
    
    r, c = samples.shape
    rand = np.random.randint(0, r)
    centers = np.zeros((k, c))
    centers[0] = samples[rand, :]   # randomly choose the first center
    
    for n in range(1, k):
        
        # distances between a point with all centers
        distances = []
        for s in samples:
            distances.append(nearest(s, centers[0:k]))
        total = np.sum(distances)
        weights = [x / total for x in distances]
        
        # select the point as new center that is far away from exist center within probability
        prob = np.random.random()
        total = 0
        x = -1
        while total < prob:
            x += 1
            total += weights[x]
        centers[n] = samples[x]
    
    return centers

# looking for the nearest distances between a point with all centers
def nearest(point, centers):
    
    min_dist = float("inf")
    
    for c in centers:
        distance  = dist_eclu(point, c)
        
        if min_dist > distance:
            min_dist = distance
    
    return min_dist


# display segmentation result using different numbers of clusters 
def display_k(images, ks, inits, coordinates = True):
    fig, axs = plt.subplots(figsize=(6 * len(images), 6), ncols = len(images))
    
    for i, img in enumerate(images):
        
        labels, centers = my_kmeans(image_encode(img, coordinates), ks[i], inits[i])
        
        # first 3 colums are encoded LAB, the last 2 columns are coordinates (x, y)
        center = np.uint8(centers[:, :3])
        out = center[labels.flatten()].reshape((img.shape))
        axs[i].imshow(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))
        axs[i].set_title("k = " + str(ks[i]))
     

# original images

img_1 = cv2.cvtColor(cv2.imread('mandm.png'), cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(cv2.imread('peppers.png'), cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(figsize=(12, 6), ncols = 2)
axs[0].imshow(img_1)
axs[1].imshow(img_2)
fig.suptitle('original images', fontsize = 18)

# Apply K-means function to color image segmentation
# using implement function with different ks
# with pixel coordinates 
        
display_k([img_1, img_2], [3, 3], ['random', 'random'])
display_k([img_1, img_2], [6, 5], ['random', 'random'])
display_k([img_1, img_2], [9, 7], ['random', 'random'])

# without pixel coordinates 

display_k([img_1, img_2], [3, 3], ['random', 'random'], False)
display_k([img_1, img_2], [6, 5], ['random', 'random'], False)
display_k([img_1, img_2], [9, 7], ['random', 'random'], False)

# using kmeans++ for initiation
# with pixel coordinates

display_k([img_1, img_2], [6, 5], ['plus', 'plus'])
display_k([img_1, img_2], [9, 7], ['plus', 'plus'])

# without pixel coordinates 

display_k([img_1, img_2], [6, 5], ['plus', 'plus'], False)
display_k([img_1, img_2], [9, 7], ['plus', 'plus'], False)





# for testing kmeans and kmeans++

"""
samples = np.random.randint(120, size = [50, 5])
print(samples)
test = kmeans_plus_init(samples, 4)
print(test)

"""

# another way for encoding images
  
"""
pixel_vector = []
    
for i in range(h):
    for j in range(w):
        pixel = [L[i, j], A[i, j], B[i, j]]
            
        # if the coordinates x, y need to be included
        if coordinates:
            pixel.extend([i, j])
                
        pixel_vector.append(pixel)
    
"""
