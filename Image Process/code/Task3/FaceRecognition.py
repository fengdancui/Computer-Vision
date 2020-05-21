# -*- coding: utf-8 -*-
"""
CLAB Task-3: Face Recognition using Eigenface
Your name (Your uniID): Fengdan Cui (u6589143)
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class FaceRecognition:
    
    def __init__(self): 
        self.size = None 
        
    # # load all images in training set, return all flatten images as a matrix, 135 x 45045
    def load_images(self, folder):
        
        data = []
        
        # read all images under folder
        for i, filename in enumerate(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, filename), 0)
            
            # store the image size
            if i == 0:
                self.size = img.shape
                
            if img is not None:
                data.append(img.flatten())
                
        return np.mat(data)     
    
    
    # perform PCA on the data matrix , return average, covariance vector, and difference
    def PCA(self, data, k):
        
        data = np.transpose(data)
        average = np.mean(data, axis = 1)   # average of all images
        diff = data - average               # differences between all images and the average
        
        eig_vals, eig_vects = np.linalg.eig(diff.T * diff)  # eigenvalues and eigenvectors
        
        eig_sort_index = np.argsort(-eig_vals)  # sort eigenvalues from large to small
        k_index = eig_sort_index[:k]            # select k eigenvectors with largest eigenvalues
        
        cov_vects = diff * eig_vects[:, k_index]
        
        return average, cov_vects, diff
    
    
    # transform the matrix into image format, 45045 x 1 -> 195 x 231
    def fit(self, img_flatten):
        return np.uint8(img_flatten.reshape(self.size))
    
    
    # three face images that are the most similar with the given image, return index of faces
    # perform nearest neighbor 
    def face_recognizer(self, test, average, cov_vects, diff_train):
        
        test = np.transpose(test)
        distances = []
        
        diff = test - average
        test_vect = cov_vects.T * diff
        
        for train in diff_train.T:
            
            train_vect = cov_vects.T * train.T
            distances.append(np.linalg.norm(test_vect - train_vect))    # calculate second norm
        
        sort_index = np.argsort(distances) 
        
        return sort_index[:3]



if __name__ == '__main__': 
    
    k = 10
    face = FaceRecognition() 
    
    train_imgs = face.load_images('Yale-FaceA/trainingset')
    average, cov_vects, diff = face.PCA(train_imgs, k)
    
    print('processing mean face ......')
    
    plt.imshow(face.fit(average), cmap = 'gray')
    plt.title('mean face')
    
    # show eigen faces
    
    print('processing eigen faces ......')
    
    col_num = 5
    row_num = np.ceil(k / col_num).astype(np.int)
    
    fig, axs = plt.subplots(figsize = (4 * col_num, 4 * row_num), ncols = col_num, nrows = row_num)
    fig.suptitle('eigen-faces ', fontsize = 18)
    
    for i in range(row_num):       
        for j in range(col_num):
            axs[i][j].imshow(face.fit(cov_vects[:, i * col_num + j]), cmap = 'gray')
    
    # find similar faces
    
    print('looking for similar faces ......')
    
    def dis_similar_face(image, data, index):
        
        fig, axs = plt.subplots(figsize = (16, 4), ncols = 4)
        axs[0].imshow(face.fit(image), cmap = 'gray')
        axs[0].set_title('test image')
        
        for i, s in enumerate(index): 
            axs[i + 1].imshow(face.fit(data[s]), cmap = 'gray')
            axs[i + 1].set_title('top ' + str(i + 1))
       
               
            
    test_imgs = face.load_images('Yale-FaceA/testset')
    
    for img in test_imgs:
        similar_index = face.face_recognizer(img, average, cov_vects, diff)
        dis_similar_face(img, train_imgs, similar_index)

    # test my own face image
    
    print('looking for similar faces with my face ......')
    
    my_face = np.mat(cv2.imread('My-Face/myface03.png', 0)).flatten()
    my_similar_index = face.face_recognizer(my_face, average, cov_vects, diff)
    dis_similar_face(my_face, train_imgs, my_similar_index)
    
    # using 144 training images
    
    print('looking for similar faces with my face in new training set ......')
    
    train_imgs_new = face.load_images('My-Face/trainingset')
    avg_new, cov_vects_new, diff_new = face.PCA(train_imgs_new, k)
    
    similar_i_new = face.face_recognizer(my_face, avg_new, cov_vects_new, diff_new)
    dis_similar_face(my_face, train_imgs_new, similar_i_new)

    
    
    
    
    
       
    
    
    
    
    
    
    
    