import os
import cv2
import numpy as np
from multiprocessing import Pool

# Camera Intrinsic Parameters (source: MSC RADAR dataset)
K1 = np.array([[639.868047808943,		0,	                     	359.848428174433],   # Left camera matrix
               [0,	            	640.607810941659,	         	277.667405577865],
               [0, 0, 1]], dtype=np.float32)

D1 = np.array([-0.249109078766160,	0.215497808485174, 0, 0, 0], dtype=np.float32)  # Left camera distortion coefficients

K2 = np.array([[638.683937622528,	  	0,	                      	364.971622592507],   # Right camera matrix
               [0,	              	639.633120192313,	          	289.163558942054],
               [0, 0, 1]], dtype=np.float32)

D2 = np.array([-0.228892193156449,	0.119049477225067, 0, 0, 0], dtype=np.float32)  # Right camera distortion coefficients

# Stereo Extrinsic Parameters
R = np.array([[0.999983103075916,	0.00489676242571525,	0.00313293479152503],
              [-0.00494222937331826,	0.999880127440026,	0.0146732790929440],
              [-0.00306070767688698,	-0.0146885148420124,	0.999887433464514]], dtype=np.float32)  # Rotation matrix
T = np.array([[-0.492828744942863], [-0.000615561667411998], [-0.00132876826743707]], dtype=np.float32)  # Translation vector

# Image resolution (adjust if needed)
image_size = (720, 540)

# Compute stereo rectification maps
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)


def rectify_images(left_folder, right_folder, output_left_folder, output_right_folder):
    """Rectify all stereo images from separate left/right folders and save results in respective output folders."""
    os.makedirs(output_left_folder, exist_ok=True)
    os.makedirs(output_right_folder, exist_ok=True)

    left_images = sorted([img for img in os.listdir(left_folder) if img.endswith(".png")])
    right_images = sorted([img for img in os.listdir(right_folder) if img.endswith(".png")])

    # Ensure image pairs exist
    image_pairs = [(l, r) for l, r in zip(left_images, right_images) if l == r]

    for left_name, right_name in image_pairs:
        left_path = os.path.join(left_folder, left_name)
        right_path = os.path.join(right_folder, right_name)

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            print(f"Skipping {left_name}, {right_name}: Unable to read images.")
            continue

        rect_left = cv2.remap(img_left, map1_x, map1_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, map2_x, map2_y, cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(output_left_folder, f"rect_{left_name}"), rect_left)
        cv2.imwrite(os.path.join(output_right_folder, f"rect_{right_name}"), rect_right)

        print(f"Processed: {left_name}, {right_name}")

    print(f"Rectified images saved to {output_left_folder} and {output_right_folder}")


# Example usage
rectify_images("test/left", "test/right", "test/rectified_left", "test/rectified_right")
