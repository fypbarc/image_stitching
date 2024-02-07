from PIL import Image

"""
for pixel in range(RANGE[0], RANGE[1], RANGE*(percentage)):
for better performance reduce the step_size for faster increase the step_size
"""

# PARAMETERS
STEP_SIZE = 0.15   # in mm
PIXEL_TO_MM = 24 # for x5 zoom (refer to excel sheet)
RANGE_PERCENTAGE = 10
RANGE_STRIDE = 1   # integer value only

# CONSTANTS / HYPERPARAMETERS
COMMON_PORTION = int(round((STEP_SIZE*1000/50)*PIXEL_TO_MM))
RANGE = (int(COMMON_PORTION*(1-RANGE_PERCENTAGE/100)), int(COMMON_PORTION*(1+RANGE_PERCENTAGE/100)))
print(COMMON_PORTION, RANGE)


# The absolute path of the folder having the images
folder_path = r"E:\College\Projects\FYP_BARC\Codes\trials_intelligent_stitch\images"
contents = os.listdir(folder_path)


def find_mse(region_1, region_2):
    """ Returns MSE value for regions given"""
    return ((region_1 - region_2)**2).mean(axis=None)
    


def find_overlap_region(image1, image2):
    """Finds the overlap regin and returns best COMMON_PORTION and pixel coordinate"""
    # Open the images using Pillow
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    
    # Ensure both images have the same dimensions
    if img1.size != img2.size:
        raise ValueError("Both images must have the same dimensions.")
    
    # Convert images to NumPy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    # Calculate the overlap width based on the overlap percentage
    width, height = img1.size

    # Initialising variables
    best_pixel = 0
    min_val = RANGE[1]
    min_mse = 1000000
    for pixel in range(RANGE[0], RANGE[1], RANGE_STRIDE):   # Through range calculated 
        overlap_region_1 = img1_array[:, -pixel:]
        overlap_region_2 = img2_array[:, :pixel]
        mse = find_mse(overlap_region_1, overlap_region_2)
        
        if mse < min_mse:
            min_val = pixel
            min_mse = mse   
    print(min_val, min_mse)     # Best found in given range


find_overlap_region(r"E:/College/Projects/FYP_BARC/Codes/trials_intelligent_stitch/images/36.50_13.40_3.48.png",
                    r"E:/College/Projects/FYP_BARC/Codes/trials_intelligent_stitch/images/36.65_13.40_3.48.png")
  
