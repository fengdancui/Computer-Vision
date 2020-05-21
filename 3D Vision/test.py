import numpy as np

points1 = np.array([
                [1,1],
                [2,3],
                [4,8],
                [3,2],
                ])
points1 = np.column_stack((points1,np.ones(points1.shape[0]))).T

H = np.array([
        [3,4,1],
        [5,6,2],
        [4,3,1]
        ])


points2 = H@points1
print(points1)
print(points2)
p1 = points1[:-1,:].T
p2 = points2[:-1,:].T

print(p1)
print(p2)

H = H.reshape((-1,1))

A_up = np.column_stack((p1,np.ones(p1.shape[0]),np.zeros((p1.shape[0],3)),-p1[:,0]*p2[:,0],-p1[:,1]*p2[:,0],-p2[:,0]))
A_below = np.column_stack((np.zeros((p1.shape[0],3)),p1,np.ones(p1.shape[0]),-p1[:,0]*p2[:,1],-p1[:,1]*p2[:,1],-p2[:,1]))

A = np.vstack((A_up,A_below))

result = np.linalg.svd(A)[-1][-1]
result = result/result[-1]
result = result.reshape((p1.shape[1]+1,-1))
print("DLT算法计算出的单应矩阵为：\n",result)
print("真实值为：\n",H.reshape((p1.shape[1]+1,-1)))