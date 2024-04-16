import numpy as np 
import cv2 as cv 
import math
import Disparity as dis
import compute_disparity_gpu
import time

L = cv.imread('./data/pictures/DSC_0005_Left.jpeg')
R = cv.imread('./data/pictures/DSC_0005_Right.jpeg')

# Converting images into grayscale
L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)

# Pre-processing by mean adjusting the images
L_gray = L_gray - np.mean(L_gray)
R_gray = R_gray - np.mean(R_gray)

print("Starting now")

# Select block size over here 
block_size = [9, 9]

# Call to CPU function 
# Un-comment this next line to run the code on the CPU
# D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)

# Measure start time
start_time = time.time()

# Call to GPU function
print(type(L_gray))
print(L_gray.shape)
D_map = compute_disparity_gpu.compute_disparity_gpu(L_gray, R_gray, block_size)

# Measure end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Time taken:", elapsed_time, "seconds")

# Smoothening the result by passing it through a median filter
D_map_filtered = cv.medianBlur(D_map, 13)

# Saving the raw and filtered disparity map
cv.imwrite('./data/raw_disparity.png', D_map)
cv.imwrite('./data/filtered_disparity.png', D_map_filtered)
