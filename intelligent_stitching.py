# Importing necessary libraries
import os
import cv2
import math
import shutil
import random
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
STEP_SIZE_X = 0.15  # in mm
STEP_SIZE_Y = 0  # in mm
PIXEL_TO_MM_X = 24  # for x2.5 zoom (refer to excel sheet)
PIXEL_TO_MM_Y = 24  # for x2.5 zoom (refer to excel sheet)
INITIAL_THRESH = 3
BIT_TRANSFORM_THRESH = 255  # pixels

# For finding overlap
RANGE_PERCENTAGE = 10  # Range of the region surrounding the COMMON_PORTION
PERCENTAGE_STRIDE = 0  # [0 to 1]  ->1 ie, steps reduce

# CONSTANTS / HYPERPARAMETERS
COMMON_PORTION_X = int(round((STEP_SIZE_X * 1000 / 50) * PIXEL_TO_MM_X))
COMMON_PORTION_Y = int(round((STEP_SIZE_Y * 1000 / 50) * PIXEL_TO_MM_Y))
RANGE_X = (int(COMMON_PORTION_X * (1 - RANGE_PERCENTAGE / 100)), int(COMMON_PORTION_X * (1 + RANGE_PERCENTAGE / 100)))
RANGE_Y = (int(COMMON_PORTION_Y * (1 - RANGE_PERCENTAGE / 100)), int(COMMON_PORTION_Y * (1 + RANGE_PERCENTAGE / 100)))
print("X: ", COMMON_PORTION_X, RANGE_X)
print("Y: ", COMMON_PORTION_Y, RANGE_Y)

# Output filename
SAVE_FILE_AS = r"concatenated.png"

# The absolute path of the folder having the images
INPUT_FOLDER_PATH = r'E:\College\Projects\FYP_BARC\Codes\trials_intelligent_stitch\images'
# We read the contents/names of all images
contents = os.listdir(INPUT_FOLDER_PATH)

if "concatenated.png" in contents:
    os.remove(os.path.join(INPUT_FOLDER_PATH, "concatenated.png"))
    contents = os.listdir(INPUT_FOLDER_PATH)


def find_mse(region_1, region_2):
    """ Returns MSE value for regions given"""
    return ((region_1 - region_2) ** 2).mean(axis=None)


def bin_to_dec(string):
    binary = int(string[2:])
    decimal, _ = 0, 0
    while binary != 0:
        dec = binary % 10
        decimal = decimal + dec * pow(2, _)
        binary = binary // 10
        _ += 1
    return decimal


def find_overlap_region(image1, image2, find: bool = True):
    """Finds the overlap regin and returns the best COMMON_PORTION and pixel coordinate"""

    # If user asks not to find overlap region we just return
    if not find:
        return COMMON_PORTION_X, None
    # Convert images to NumPy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Initialising variables
    # best_pixel = 0
    min_val = [0]
    min_mse = [255]
    # Loops through each value in range to find the best common area
    _step = int((RANGE_X[1] - RANGE_X[0]) * PERCENTAGE_STRIDE)
    if _step < 1:
        _step = 1
    for p in range(RANGE_X[0], RANGE_X[1], _step):
        overlap_region_1 = img1_array[:, -p:]
        overlap_region_2 = img2_array[:, :p]
        _mse = find_mse(overlap_region_1, overlap_region_2)
        if _mse < min_mse:
            min_val[0] = p
            min_mse[0] = _mse
    # Best found in given range
    # print(min_val, min_mse)
    return min_val[0], min_mse[0]


class MergeStitchedArea:
    """Applies filter on stitched portion depending on their intensity difference
        pre, post: are columns
        # diff < 128   or diff > 128
        # <: diff/2  is range
        # >: (255-diff)/2 is range"""

    def __init__(self, pre_stitch_col, post_stitch_col):
        self.pre = pre_stitch_col
        self.post = post_stitch_col
        self.difference = np.zeros(np.shape(self.post))
        self.new_diff = np.zeros(np.shape(self.post))

    def check_first_difference(self):
        """Checks the initial difference and conditions with INITIAL_THRESH to decide whether to go ahead with further processing or not"""
        for k in range(len(self.pre)):
            self.difference[k] = np.int16(self.post[k]) - np.int16(self.pre[k])
            # print("old", self.post[k], self.pre[k], self.difference[k])
            # If INITIAL_THRESH is more than difference between 2 neighbouring pixels we do nothing
            if abs(self.difference[k]) <= INITIAL_THRESH:
                self.new_diff[k] = None
                continue
            # Get range of pre- and post-pixels for expanding them later to reduce the intensity difference4
            pre_range: int  # Just declaring but not assigning
            post_range: int
            if self.pre[k] <= 128:
                pre_range = int(self.pre[k] / 2)
            else:
                pre_range = int((255 - self.pre[k]) / 2)

            if self.post[k] <= 128:
                post_range = int(self.post[k] / 2)
            else:
                post_range = int((255 - self.post[k]) / 2)
            # print("pre-pixel range and post-pixel range: ", pre_range, post_range)

            if self.difference[k] > 0:
                self.pre[k] += min(self.difference[k] / 2, pre_range / 2)
                self.post[k] -= min(self.difference[k] / 2, post_range / 2)
            elif self.difference[k] < 0:
                self.pre[k] -= min(abs(self.difference[k] / 2), pre_range / 2)
                self.post[k] += min(abs(self.difference[k] / 2), post_range / 2)
            else:
                pass

            self.new_diff[k] = np.int16(self.post[k]) - np.int16(self.pre[k])

            # print("new", self.post[k], self.pre[k], self.new_diff[k], k)

            # if new_diff > BIT_TRANSFORM_THRESH:
            #     pre_bin = bin(pre[i])
            #     post_bin = bin(post[i])
            #
            #     for i in range(len(pre_bin[2:])):
            #         if pre_bin[i] != post_bin[i]:
            #             maxx = max(int(pre_bin[i]), int(post_bin[i+1]))
            #             pre_bin[i+1] = str(maxx)
            #             pre_bin[i+1] = post_bin[i]
            #     return pre, post
            # else:

def rename_files(folder_path=None):
    """This function will rename all files inside the source_image folder and save it as x_y coordinates as wanted  by user"""
    if folder_path is None:
        return

    files_list = os.listdir(folder_path)

    random_name = random.choice(files_list)
    terms_in_name = random_name.split('_')
    terms_in_name[-1] = terms_in_name[-1][:-4]

    print("Here is an example filename: ", random_name)
    rename = input("Do you want to rename files? to eg,Xcoor_Ycoor.png [Y or N]: ").lower()
    if rename == 'n':
        return

    print(f"Enter the axis index in og name between 1-{len(terms_in_name)}: ")
    x_index = int(input("X-axis: ")) - 1
    y_index = int(input("Y-axis: ")) - 1

    for k in range(len(files_list)):
        if '_' in files_list[k]:
            current_path = os.path.join(folder_path, files_list[k])
            terms_in_name = files_list[k].split('_')
            terms_in_name[-1] = terms_in_name[-1][:-4]
            new_name = str(terms_in_name[x_index] + "_" + terms_in_name[y_index] + ".png")
            new_path = os.path.join(folder_path, new_name)
            # print(new_path)
            shutil.move(current_path, new_path)


# We split X-range and Y-range to obtain all unique values
image_dict = dict()
# We first rename files in the folder since our code is sensitive to file names
rename_files(folder_path=INPUT_FOLDER_PATH)
# We change the file names up to 2 decimal places
y_range = sorted(set([f"{float(y.split('_')[-1][:-4]):.2f}" for y in contents]))
# print(y_range)

# We obtain all values of X for a particular value of Y
prev_img = None
for y_coor in y_range:
    # We store values of all X's for each unique Y values
    x_for_y = sorted(set([f"{float(x.split('_')[0]):.2f}" for x in contents if str(y_coor) in x[:-4]]))
    # We iterate through all possible values of x for each value of Y
    for i in range(len(x_for_y)):
        # Iterates to get all the images
        path = os.path.join(INPUT_FOLDER_PATH, str(x_for_y[i] + "_" + y_coor + ".png"))
        # print(path)
        # print(os.path.basename(path))
        # print(path)
        if i == 0:
            # If first image is on roll the do nothing
            base_img = cv2.imread(path, 0)
            prev_img = base_img
        else:
            img = cv2.imread(path, 0)
            pixel, mse = find_overlap_region(prev_img, img, find=False)
            # We slice the image and get the meaningful part ie, uncommon region of current image
            # We concat the current sliced to previous image
            concatenated_image = np.hstack([prev_img, img[:, -pixel:]])
            # We apply filters over stitched region to reduce discreteness in image
            pre_stitch = concatenated_image[:, -(pixel-1)]  # Obtain the post pixel column of stitched image
            post_stitch = concatenated_image[:, -pixel]  # Obtain the pre pixel column of stitched image
            # Get post-processed post and pre pixel columns and replace in concatenated image
            MergeStitchedArea(pre_stitch, post_stitch).check_first_difference()
            concatenated_image[:, -(pixel-1)] = pre_stitch
            concatenated_image[:, -pixel] = post_stitch
            # We use the current image as parent image in the next loop
            prev_img = concatenated_image
    # We store the concatenated image for every y value in a dictionary and further concat it in Y direction
    image_dict[f"{float(y_coor)}"] = prev_img

row_images = list(image_dict.values())
for i in range(len(row_images)):
    if i == 0:
        base_img = row_images[0]
        # print(np.shape(base_img))
        prev_img = base_img
    else:
        img = row_images[i][-COMMON_PORTION_Y:, :]
        # We concat the current sliced to previous image
        concatenated_image = np.vstack([prev_img, img])
        # We use the current image as parent image in the next loop
        prev_img = concatenated_image

# Prints the path of image which and where it's being saved
print(os.path.join(INPUT_FOLDER_PATH, f"{SAVE_FILE_AS}"))
cv2.imwrite(os.path.join(INPUT_FOLDER_PATH, f"{SAVE_FILE_AS}"), concatenated_image)
