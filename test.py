import cv2
import numpy as np

im = cv2.imread("Hello2.jpg")
array = np.array(im)
print(array)
print(array.shape)


# import os
# os.rename(r"Hello.jpg", r"Hello2.jpg")
