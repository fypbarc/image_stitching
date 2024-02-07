import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

left_value = 100
right_value = 200

image = np.zeros((600, 600), dtype=np.uint8)
boundary = 200  


for i in range(0, 155, 1):
    print(i)
    left_value += 1
    right_value -= 1

    # Assign values to image
    image[:, :boundary] = 80
    image[:, -boundary:] = 160

    # Display the image (consider displaying outside the loop for final version)
    cv2.imshow("Grayscale Image", image)
    cv2.waitKey(500) 


cv2.destroyAllWindows()
