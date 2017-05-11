import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.329, 0.587, 0.114])

img = mpimg.imread('test.png')
gray = rgb2gray(img)
print(gray.reshape(28,28,1))
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.imsave("2.png",gray)
