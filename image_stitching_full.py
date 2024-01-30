# Importing necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE_X = 2   # in mm
STEP_SIZE_Y = 0   # in mm
PIXEL_TO_MM_X = 24 # for x2.5 zoom (refer to excel sheet)
PIXEL_TO_MM_Y = 24 # for x2.5 zoom (refer to excel sheet)

COMMON_PORTION_X = int(round((STEP_SIZE_X*1000/50)*PIXEL_TO_MM_X))
COMMON_PORTION_Y = int(round((STEP_SIZE_Y*1000/50)*PIXEL_TO_MM_Y))

#print(COMMON_PORTION_X, COMMON_PORTION_Y)
SAVE_FILE_AS = r"concatenated.png"

# The absolute path of the folder having the images
folder_path = r'E:\College\Projects\FYP_BARC\_CAMERA_IMAGES_\2.50\2.50'
# A temporary variable
prev_img = None
# We read the contents/names of all images
contents = os.listdir(folder_path)
#print(contents)

# We split X-range and Y-range to obtain all unique values
image_dict = dict()
y_range = sorted(set([(y.split("_")[1]).split(".")[0] for y in contents]))
#print(y_range)

# We obtain all values of X for a particular value of Y
for y_coor in y_range:
    x_for_y = sorted(set([x for x in contents if y_coor in x.split('_')[1]]))
    
    # We iterate through all possibile values of x for each value of Y
    for i in range(len(x_for_y)):
    # Iterates to get all of the images
        path = os.path.join(folder_path, x_for_y[i])
        #print(path)
        if i == 0:
            # If first image is on roll the do nothing
            base_img = cv2.imread(path)
            prev_img = base_img
        else:
            # We slice the image and get the meaningful part ie, uncommon region of current image
            img = (cv2.imread(path))[:, -COMMON_PORTION_X:]
            # We concat the current sliced to previous image    
            concatenated_image = np.hstack([prev_img, img])    
            # We use the current image as parent image in the next loop
            prev_img = concatenated_image
            
    # We store the concatenated image for every y value in a dictionary and further concat it in Y direction
    image_dict[f"{float(y_coor)}"] =  concatenated_image

row_images = list(image_dict.values())
for i in range(len(row_images)):
    if i == 0:
        base_img = row_images[0]
        #print(np.shape(base_img))
        prev_img = base_img
    else:
        img = row_images[i][-COMMON_PORTION_Y:, :]
        # We concat the current sliced to previous image
        concatenated_image = np.vstack([prev_img, img])
        # We use the current image as parent image in the next loop
        prev_img = concatenated_image
# plt.imshow(prev_img)
# plt.show()

# Prints the path of image which and where it's being saved
print(os.path.join(folder_path, f"{SAVE_FILE_AS}"))
cv2.imwrite(os.path.join(folder_path, f"{SAVE_FILE_AS}"), concatenated_image)
