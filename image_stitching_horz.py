# Importing necessary libraries
import os
import cv2
import numpy as np

STEP_SIZE = 2   # in mm
PIXEL_TO_MM = 24 # for x5 zoom (refer to excel sheet)

COMMON_PORTION = int(round((STEP_SIZE*1000/50)*43.25))
print(COMMON_PORTION)
SAVE_FILE_AS = r"concatenated.png"

# The absolute path of the folder having the images
folder_path = r"E:\College\Projects\FYP_BARC\_CAMERA_IMAGES_\2.50\2.50"
contents = os.listdir(folder_path)

# A temporary variable
prev_img = None
for i in range(len(contents)):
    # Iterates to get all of the images
    path = os.path.join(folder_path, contents[i])
    
    if i == 0:
        # If first image is on roll the do nothing
        base_img = cv2.imread(path)
        prev_img = base_img
    else:
        # We slice the image and get the meaningful part ie, uncommon region of current image
        img = (cv2.imread(path))[:, -COMMON_PORTION:]
        # Since concat works only on same sized images we add a black image of the same resolution as the common region which is sliced in the above area
        zeros_to_add = np.zeros((1024, COMMON_PORTION, 3), dtype=np.uint8)
        result = np.hstack((img, zeros_to_add))
        # We concat the current sliced to previous image
        concatenated_image = cv2.hconcat([prev_img, result])
        concatenated_image = concatenated_image[:, :-COMMON_PORTION]
        # We use the current image as parent image in the next loop
        prev_img = concatenated_image
# Prints the path of image which and where it's being saved
print(os.path.join(folder_path, f"{SAVE_FILE_AS}"))
cv2.imwrite(os.path.join(folder_path, f"{SAVE_FILE_AS}"), concatenated_image)
        

