import numpy as np 
import cv2 as cv 
import math
import Disparity as dis
import host_code
import time

vid_folder = "../data/videos/"
file = "IMG_9036"

left_cap = cv.VideoCapture(f"{vid_folder}{file}_Left.mp4")
right_cap = cv.VideoCapture(f"{vid_folder}{file}_Right.mp4")
output_filename = f"{vid_folder}output/{file}_disparity_output.avi"
output_filtered_filename = f"{vid_folder}output/{file}_disparity_filtered_output.avi"
framerate = 20.0 # todo: calculated value from run speed

# Check if video captures are opened successfully
if not left_cap.isOpened() or not right_cap.isOpened():
    print("Error opening video streams or files")

# Define video writer object for disparity output
fourcc = cv.VideoWriter_fourcc(*'XVID')  # video format

# read in first frame of each side to use for output shape
ret_left, left_frame = left_cap.read()
ret_right, right_frame = right_cap.read()
if not ret_left:
    print("No frame received from left video stream. Exiting...")
    exit()
cv.imshow('Disparity Map Filtered', right_frame)
width, height = left_frame.shape[:2] # output same dimensions as input left
out = cv.VideoWriter(output_filename, fourcc, framerate, (width, height))
out_filtered = cv.VideoWriter(output_filtered_filename, fourcc, framerate, (width, height))

# Check if video captures are opened successfully
if not left_cap.isOpened() or not right_cap.isOpened():
    print("Error opening video streams or files")

# Define font for FPS display
font = cv.FONT_HERSHEY_SIMPLEX

# Variables for FPS calculation
start_time = 0
frames_processed = 0

print("Starting video processing...")

# Loop through each frame of the video
while True:
    # get frame
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    # Check if frames are read correctly
    if not ret_left or not ret_right:
        print("No frames received from video streams. Exiting...")
        break

    # Converting images into grayscale
    L_gray = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
    R_gray = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)

    # Pre-processing by mean adjusting the images
    L_gray = L_gray - np.mean(L_gray)
    R_gray = R_gray - np.mean(R_gray)

    print("Starting now")

    # Select block size over here 
    block_size = [9, 9]

    # Start time for FPS calculation (on the first frame)
    if frames_processed == 0:
        start_time = time.time()

    # Call to CPU function 
    # Un-comment this next line to run the code on the CPU
    # D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)

    # Call to GPU function
    D_map = host_code.compute_disparity_gpu(L_gray, R_gray, block_size)

    # Smoothening the result by passing it through a median filter
    D_map_filtered = cv.medianBlur(D_map, 13)

    # Calculate FPS
    frames_processed += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 0.5:  # Update FPS every 0.5 second
        fps = frames_processed / elapsed_time
        frames_processed = 0
        start_time = current_time

        # Put FPS text on the frame
        fps_text = f"FPS: {int(fps)}"
        cv.putText(D_map_filtered, fps_text, (10, 30), font, 1, (0, 0, 255), 2)

    print("Time taken:", elapsed_time, "seconds") # todo remove to reduce IO

    # Write the disparity frame to the output video
    out.write(D_map)
    out_filtered.write(D_map_filtered)

    # Show the disparity map (optional)
    # cv.imshow('Disparity Map Filtered', D_map_filtered)
    # cv.waitKey(1)  # Adjust wait time as needed

# Release resources
left_cap.release()
right_cap.release()
out.release()
out_filtered.release()
cv.destroyAllWindows()

print("Video processing complete!")
