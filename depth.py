import cv2
import numpy as np
import argparse
import os
import glob

def compute_disparity(left_rectified, right_rectified, num_disparities=64, block_size=11, window_size=11):
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=block_size, 
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    return disparity

def calc_depth_map(disparity, focal_length, baseline):
    disparity[disparity <= 0] = 1e-3  # division by zero or negative disparities
    depth_map = np.single(focal_length * baseline) / np.single(disparity)
    return depth_map

def process_stereo(left_dir, right_dir, output_dir):
    # Get image file list
    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
    os.makedirs(output_dir, exist_ok=True)

    # Camera parameters (hardcoded for MSC radar dataset)
    K1 = np.array([[646.96, 0, 374.84], [0, 647.31, 274.11], [0, 0, 1]])
    D1 = np.array([-0.233, 0.142, 0, 0, 0])
    K2 = np.array([[647.67, 0, 367.69], [0, 647.67, 285.20], [0, 0, 1]])
    D2 = np.array([-0.232, 0.129, 0, 0, 0])
    R = np.array([[0.999995, -0.0018, -0.00243], [0.0018, 0.999998, 0.00042], [0.00243, -0.00043, 0.999997]])
    T = np.array([-497.79, -2.18, 3.78])

    # Process all image pairs
    for left_path, right_path in zip(left_images, right_images):
        # Load images
        left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
        image_size = left_image.shape[::-1]
    
        # Undistort and rectify images
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)
        left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
    
        # Compute disparity
        disparity_map = compute_disparity(left_rectified, right_rectified, num_disparities=64, block_size=11, window_size=11)
    
        # Compute depth map
        focal_length = K1[0, 0]  # In pixels
        baseline = np.linalg.norm(T)  # In mm
        depth_map = calc_depth_map(disparity_map, focal_length, baseline)
        depth_map = np.clip(depth_map, 0, 25000)  # Clip depth range
    
        # Normalize and save results
        disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        filename = os.path.basename(left_path)
        output_path = os.path.join(output_dir, filename)
    
        cv2.imwrite(output_path, depth_normalized)
        
    print(f"Saved depth maps in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo depth estimation: SGBM (OpenCV).")
    parser.add_argument("left", help="Path to left directory")
    parser.add_argument("right", help="Path to right directory")
    parser.add_argument("output", help="Output directory for results")
    args = parser.parse_args()
    
    process_stereo(args.left, args.right, args.output)
