import cv2
import numpy as np
import glob
import pickle
import os

# Chessboard size (inner corners: 8x6 for a 9x7 grid of squares)
CHECKERBOARD = (8, 6)  # 8 columns, 6 rows of inner corners
SQUARE_SIZE = 0.025  # Size of each square in meters (25mm = 0.025m)

# Prepare object points (3D points of chessboard corners in real world)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Load images from captured_images folder
images = glob.glob('captured_images/image_*.jpg')  # Matches image_01.jpg to image_19.jpg
print(f"Found {len(images)} images: {images}")  # Debug: Verify images are loaded

# Create a folder to save images with detected corners
if not os.path.exists('corners_detected'):
    os.makedirs('corners_detected')

# Criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Variable to store image dimensions
img_shape = None

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image to improve corner detection
    gray = cv2.equalizeHist(gray)  # Enhance contrast

    # Store image dimensions
    if img_shape is None:
        img_shape = gray.shape[::-1]  # (width, height)

    # Find chessboard corners with additional flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=flags)

    if ret:
        print(f"Corners detected in {fname}")
        objpoints.append(objp)
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and save corners for verification
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imwrite(f'corners_detected/corners_{idx:02d}.jpg', img)
    else:
        print(f"No corners detected in {fname}")

# Check if we have enough data for calibration
if not objpoints or not imgpoints:
    raise ValueError("No chessboard corners detected in any images. Check your images or chessboard pattern.")

# Calibrate camera using stored image dimensions
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

# Save calibration parameters
calibration_data = {
    'camera_matrix': mtx,
    'distortion_coefficients': dist
}
with open('calibration_data.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

# Print results
print("Camera Matrix (K):\n", mtx)
print("Distortion Coefficients:\n", dist)

# Test undistortion on the last image
img = cv2.imread(images[-1])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite('undistorted_last.jpg', undistorted)