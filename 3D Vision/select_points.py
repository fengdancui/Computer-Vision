from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


img1 = Image.open('Left.jpg')
img2 = Image.open('Right.jpg')

# plt.imshow(img1)
# uv1 = plt.ginput(6, timeout = 0) # Graphical user interface to get 6 points
# np.save('uvLeft', uv1)
# print(uv1)

plt.imshow(img2)
uv2 = plt.ginput(6, timeout = 0) # Graphical user interface to get 6 points
np.save('uvRight', uv2)

print(uv2)