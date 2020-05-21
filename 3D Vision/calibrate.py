# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vgg_KR_from_P import vgg_KR_from_P
import cv2
import math

# I = Image.open('stereo2012a.jpg')
# plt.imshow(I)
# uv = plt.ginput(6, timeout = 0) # Graphical user interface to get 6 points
# XYZ = np.array([[7, 7, 0], [21, 28, 0], [14, 0, 21],
#                 [21, 0, 7], [0, 35, 21], [0, 21, 7]])
# np.save('uv', uv)
# np.save('XYZ', XYZ)
# print(uv)
# print(XYZ)

#####################################################################
def calibrate(im, XYZ, uv):
    # TBD

    r, c = XYZ.shape
    X, Y, Z = np.hsplit(XYZ, 3)
    u, v = np.hsplit(uv, 2)

    # generate matrix A with shape (12, 11)
    # [X, Y, Z, 1, 0, 0, 0, 0, -Xu, -Yu, -Zu]
    # [0, 0, 0, 0, X, Y, Z, 1, -Xv, -Yv, -Zv]
    A_up = np.column_stack((XYZ, np.ones(r), np.zeros((r, 4)), -X * u, -Y * u, -Z * u))
    A_down = np.column_stack((np.zeros((r, 4)), XYZ, np.ones(r), -X * v, -Y * v, -Z * v))
    A = np.vstack((A_up, A_down))

    # generate matrix b with shape (12, )
    # [u1, u2, u3, u4, u5, u6, v1, v2, v3, v4, v5, v6].T
    b = np.vstack((u, v)).reshape(12)

    # least squares to calculate q
    ls = np.linalg.lstsq(A, b, rcond=None)
    q = ls[0]

    # reshape L to 2x4 camera matrix
    C = np.hstack((q, 1)).reshape((3, 4))
    print('The error in satisfying the camera calibration matrix constraints is: %f' % ls[1][0])

    # according to C, calculate projected image coordinates
    proj_uv = coord_trans(C, XYZ)

    # calculate error between projected coordinates and original uv
    error = np.linalg.norm(uv - proj_uv)
    print(
        'The mean squared error between the positions of the uv coordinates and the projected XYZ coordinates is: %f' % error)

    # draw projected image coordinates from XYZ
    plt.plot(*zip(*proj_uv), 'o', color='b')

    # original collected image coordinates
    plt.plot(*zip(*uv), 'b+', color='r')

    # draw XYZ world coordinate system
    world_axis = [[0, 0, 0], [21, 0, 0], [0, 21, 0], [0, 0, 21]]
    label = ['x', 'y', 'z']
    img_axis = coord_trans(C, world_axis)
    for i, axis in enumerate(img_axis[1:]):
        plt.annotate(label[i - 1],
                     xy=img_axis[0],
                     xytext=axis,
                     color='r',
                     arrowprops=dict(arrowstyle="<-", color="r"))

    plt.imshow(im)
    plt.show()

    return C

# project a list of world coordinates to image coordinates based on camera matrix C
def coord_trans(C, world):
    img_coord = []
    for w in world:
        h_uv = np.dot(C, np.hstack((w, 1)))

        # Convert homogeneous coordinates to non-homogeneous coordinates
        img_coord.append(h_uv[:2] / h_uv[2])
    return img_coord


'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% Fengdan Cui on 14, May 2020
'''


############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):

    rows = len(uBase)

    # generate A with shape (12, 9)
    # [u, v, 1, 0, 0, 0, -uu', -vu', -u']
    # [0, 0, 0, u, v, 1, -uv', -vv', -v]
    A_up = np.column_stack((uBase, vBase, np.ones(rows), np.zeros((rows, 3)), -uBase * u2Trans, -vBase * u2Trans, -u2Trans))
    A_down = np.column_stack((np.zeros((rows, 3)), uBase, vBase, np.ones(rows), -uBase * v2Trans, -vBase * v2Trans, -v2Trans))
    A = np.vstack((A_up, A_down))

    # SVD and select the most right column of V
    result = np.linalg.svd(A)[-1][-1]

    # reshape column with shape of (9, ) to (3, 3)
    H = (result / result[-1]).reshape((3, 3))

    return H


'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% Fengdan Cui 20, May
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q, r] = np.linalg.qr(A.T)  # numpy has QR decomposition, here we can do it
    # with Q: orthonormal and R: upper triangle. Apply QR
    # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R, Q


# main section
# camera calibrate


uv = np.load('uv.npy')
XYZ = np.load('XYZ.npy')
uv_left = np.load('uvLeft.npy')
uv_right = np.load('uvRight.npy')

img = cv2.cvtColor(cv2.imread('stereo2012a.jpg'), cv2.COLOR_BGR2RGB)
C = calibrate(img, XYZ, uv)

# Decompose the C matrix into K, R, t
K, R, t = vgg_KR_from_P(C)
print('The internal calibration parameters are: ')
print(K)
print('The rotation parameters are: ')
print(R)
print('The translation parameters are: ')
print(t)

# calculate the pitch angle of the camera with respect to the X-Z plane
angle = math.degrees(math.atan(K[0][2] / K[1][2]))
print('The pitch angle of the camera with respect to the X-Z plane is: %f' % angle)

# homography

img_left = cv2.cvtColor(cv2.imread('Left.jpg'), cv2.COLOR_BGR2RGB)
img_right = cv2.cvtColor(cv2.imread('Right.jpg'), cv2.COLOR_BGR2RGB)

H = homography(uv_right[:, 0], uv_right[:, 1], uv_left[:, 0], uv_left[:, 1])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(img_left)
ax1.plot(*zip(*uv_left), 'b+', color='r')

uv_input = np.column_stack((uv_left, np.ones(len(uv_left))))
uv_out = np.dot(H, uv_input.T)
uv_out = (uv_out[:-1] / uv_out[-1]).T
error = np.sum(np.power(uv_right - uv_out, 2))
print('The error after swapping Left is: %f ' % error)

ax2.imshow(img_right)
ax2.plot(*zip(*uv_out), 'o', color='b')
ax2.plot(*zip(*uv_right), 'b+', color='r')

trans_img = np.zeros(img_left.shape)
r, c, _ = img_left.shape
for i in range(r):
    for j in range(c):
        uv = [i, j, 1]
        out = np.array(np.mat(H) * np.mat(uv).T)
        x, y = (out[:-1] / out[-1]).astype(np.int)
        if x < img_right.shape[0] and y < img_right.shape[1] and x > 0 and y > 0:
            trans_img[x, y, :] = img_right[x, y, :]

ax3.imshow(np.uint8(trans_img))
plt.show()


# another way for generating A of camera calibrate

# for i, world in enumerate(XYZ):
#     # caculate 2d matrix [[−Xu, −Yu, −Zu,], [−Xv, −Yv, -Zv]]
#     # = -[u, v].T * [X, Y, Z]
#     pixel = uv[i].reshape((len(uv[i]), 1))
#     coord = world.reshape((1, len(world)))
#     p_mul_c = -np.dot(pixel, coord)
#
#     # generate 2d matrix [[X, Y, Z, 1], [0, 0, 0, 0]]
#     # and [[0, 0, 0, 0], [X, Y, Z, 1]]
#     homog_coord = np.hstack((world, 1))
#     zero = np.zeros(len(homog_coord))
#     ho_up_zero = np.vstack((zero, homog_coord))
#     ho_down_zero = np.vstack((homog_coord, zero))
#
#     # concatenate three matrix to generate 2x11 matrix
#     # the shape of final A is 12x11 (12 = 2x6points)
#     A.extend(np.column_stack((ho_down_zero, ho_up_zero, p_mul_c)))
#     # the shape of final A is 12x1
#     b.extend(uv[i].T)

# another method for generating A

# for i in range(len(u2Trans)):
#     ho_uv = [uBase[i], vBase[i], 1]
#     zero = np.zeros(len(ho_uv))
#     ho_up_zero = -np.vstack((zero, ho_uv))
#     ho_down_zero = -np.vstack((ho_uv, zero))
#
#     t_mul_b = np.array(np.mat([u2Trans[i], v2Trans[i]]).T * np.mat(ho_uv))
#     A.extend(np.column_stack((ho_down_zero, ho_up_zero, t_mul_b)))
#
# print(np.array(A).shape)


