import cv2
import numpy as np

# Load calibration data
data = np.load("3D-Reconstruction/calibration_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Load the distorted image
img = cv2.imread("3D-Reconstruction/room_annotation/room.png")

# Undistort the image
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)
undistorted_img = cv2.undistort(
    img, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# Save the undistorted image
cv2.imwrite("3D-Reconstruction/room_annotation/undistorted_room.png", undistorted_img)

# Display the original and undistorted images
cv2.imshow("Distorted Image", img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
