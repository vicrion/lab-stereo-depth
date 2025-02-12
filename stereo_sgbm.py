import cv2
import numpy as np
import argparse

def compute_disparity(left_img_path, right_img_path, min_disp, num_disp, block_size, uniqueness, speckle_win, speckle_range):
    # Load left and right images
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        print("Error: One or both images not found.")
        return
    
    # Create StereoSGBM matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_win,
        speckleRange=speckle_range,
        disp12MaxDiff=1,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2
    )
    
    # Compute disparity
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    
    # Normalize for visualization
    disparity_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_visual = np.uint8(disparity_visual)
    
    # Show the result
    Hori = np.concatenate((left_img, disparity_visual), axis=1)
    cv2.putText(Hori, f"min disp={min_disp}, block sz={block_size}, uniqueness={uniqueness}, speck sz={speckle_win}, speck rg={speckle_range}",
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2, cv2.LINE_AA)
    # cv2.namedWindow("Disparity Map", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Disparity Map", Hori)
    wait_time = 1000
    while cv2.getWindowProperty('Disparity Map', cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute disparity map using StereoSGBM")
    parser.add_argument("left_img", help="Path to the left image")
    parser.add_argument("right_img", help="Path to the right image")
    parser.add_argument("--min_disp", type=int, default=0, help="Minimum disparity")
    parser.add_argument("--num_disp", type=int, default=64, help="Number of disparities (must be divisible by 16)")
    parser.add_argument("--block_size", type=int, default=9, help="Block size for matching (odd number)")
    parser.add_argument("--uniqueness", type=int, default=10, help="Uniqueness ratio")
    parser.add_argument("--speckle_win", type=int, default=100, help="Speckle window size")
    parser.add_argument("--speckle_range", type=int, default=32, help="Speckle range")
    
    args = parser.parse_args()
    
    compute_disparity(args.left_img, args.right_img, args.min_disp, args.num_disp, args.block_size, args.uniqueness, args.speckle_win, args.speckle_range)
