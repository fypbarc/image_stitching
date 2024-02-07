# Importing necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
for pixel in range(RANGE[0], RANGE[1], RANGE*(percentage)):
for better performance reduce the step_size for faster increase the step_size

pre_range, post_range and replace by peak range (val-range/2)value to minimise difference
if diff is eg 255 we take avg of 3x3 matrix to replace and check again else avg 5x5 then replace by peak

if replace by peak doesnt help then the first bit which differs and use mam algo to decide the bit 
"""

# PARAMETERS
STEP_SIZE_X = 2   # in mm
STEP_SIZE_Y = 0   # in mm
PIXEL_TO_MM_X = 24 # for x2.5 zoom (refer to excel sheet)
PIXEL_TO_MM_Y = 24 # for x2.5 zoom (refer to excel sheet)
INITIAL_THRESH = 20
BIT_TRANSFORM_THRESH = 20  # pixels

# For finding overlap
RANGE_PERCENTAGE = 10   # Range of the region surrounding the COMMON_PORTION
PERCENTAGE_STRIDE = 1   # [0 to 1]  ->1 ie, steps reduce

# CONSTANTS / HYPERPARAMETERS
COMMON_PORTION_X = int(round((STEP_SIZE_X*1000/50)*PIXEL_TO_MM_X))
COMMON_PORTION_Y = int(round((STEP_SIZE_Y*1000/50)*PIXEL_TO_MM_Y))
RANGE_X = (int(COMMON_PORTION_X*(1-RANGE_PERCENTAGE/100)), int(COMMON_PORTION_X*(1+RANGE_PERCENTAGE/100)))
RANGE_y = (int(COMMON_PORTION_Y*(1-RANGE_PERCENTAGE/100)), int(COMMON_PORTION_Y*(1+RANGE_PERCENTAGE/100)))
print("X: ",COMMON_PORTION_X, RANGE_X)
print("Y: ",COMMON_PORTION_X, RANGE_X)

# Ourput filename
SAVE_FILE_AS = r"concatenated.png"

# The absolute path of the folder having the images
folder_path = r'E:\College\Projects\FYP_BARC\Codes\trials_intelligent_stitch\images'
# We read the contents/names of all images
contents = os.listdir(folder_path)


def find_mse(region_1, region_2):
    """ Returns MSE value for regions given"""
    return ((region_1 - region_2)**2).mean(axis=None)

def bin_to_dec(string):
    binary = int(string[2:])
    decimal, i = 0, 0
    while (binary != 0):
        dec = binary%10
        decimal = decimal + dec*pow(2, i)
        binary = binary//10
        i += 1
        return binary
    

def find_overlap_region(image1, image2):
    """Finds the overlap regin and returns best COMMON_PORTION and pixel coordinate"""
    # Convert images to NumPy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Initialising variables
    best_pixel = 0
    min_val = RANGE_X[1]
    min_mse = 1000000
    # Loops through each value in range to find best common area
    for pixel in range(RANGE_X[0], RANGE_X[1], int((RANGE_X[1]-RANGE_X[0])*PERCENTAGE_STRIDE)):    
        overlap_region_1 = img1_array[:, -pixel:]
        overlap_region_2 = img2_array[:, :pixel]
        mse = find_mse(overlap_region_1, overlap_region_2)
        
        if mse < min_mse:
            min_val = pixel
            min_mse = mse
    # Best found in given range
    return min_val, min_mse


def merge_stitched_area(pre, post):
    """Applies filter on stitched portion depending on their intensity difference
    pre, post: are columns"""
    # diff < 128   or diff > 128
    # <: diff/2  is range
    # >: (255-diff)/2 is range
    for i in range(len(pre)):
        print(pre[i], post[i])
        difference = post[i]-pre[i]

        pre_range = 0
        post_range = 0
        
        if pre[i] <= 128:
            pre_range = int(pre[i]/2)
        else:
            pre_range = int((255-pre[i])/2)
            
        if post[i] <= 128:
            post_range = int(post[i]/2)
        else:
            post_range = int((255-post[i])/2)
            
        print(pre_range, post_range)
        
        if abs(difference) <= INITIAL_THRESH:
            return
        
        if difference > 0:
            pre[i] += min(difference/2, pre_range/2)
            post[i] -= min(difference/2, post_range/2)
        elif difference < 0:
            pre[i] -= min(difference/2, pre_range/2)
            post[i] += min(difference/2, post_range/2)

        new_diff = post[i]-pre[i]
        print(difference, new_diff, pre[i], post[i])

        if new_diff > BIT_TRANSFORM_THRESH:
            pre_bin = bin(pre[i])
            post_bin = bin(post[i])

            for i in range(len(pre_bin[2:])):
                if pre_bin[i] != post_bin[i]:
                    pre_bin[i+1] = str(max(int(pre_bin[i]), int(post_bin[i+1]))) = post_bin[i]

                pre
            
        

# We split X-range and Y-range to obtain all unique values
image_dict = dict()
y_range = sorted(set([(y.split("_")[1]).split(".")[0] for y in contents]))
#print(y_range)

# We obtain all values of X for a particular value of Y
prev_img = None
for y_coor in y_range:
    x_for_y = sorted(set([x for x in contents if y_coor in x.split('_')[1]]))
    
    # We iterate through all possibile values of x for each value of Y
    for i in range(len(x_for_y)):
    # Iterates to get all of the images
        path = os.path.join(folder_path, x_for_y[i])
        #print(path)
        if i == 0:
            # If first image is on roll the do nothing
            base_img = cv2.imread(path, 0)
            prev_img = base_img
        else:
            img = cv2.imread(path, 0)
            pixel, mse = find_overlap_region(prev_img, img)
            # We slice the image and get the meaningful part ie, uncommon region of current image
            # We concat the current sliced to previous image    
            concatenated_image = np.hstack([prev_img, img[:, pixel:]])
            # We apply filters over stitched region to reduce discreteness in image
            pre_stitch = concatenated_image[:, pixel-1]
            post_stitch = concatenated_image[:, pixel]
            merged_column = merge_stitched_area(pre_stitch, post_stitch)
            concatenated_image[:, pixel] = merged_column
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
