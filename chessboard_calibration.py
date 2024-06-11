import cv2
import numpy as np
import glob


def draw_reprojected_points(images, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # Draw the original detected corners
        for point in imgpoints[i]:
            x, y = point.ravel()
            cv2.circle(
                img, (int(x), int(y)), 5, (0, 0, 255), -1
            )  # Red points for original detected corners

        # Draw the reprojected corners
        for point in imgpoints2:
            x, y = point.ravel()
            cv2.circle(
                img, (int(x), int(y)), 5, (0, 255, 0), -1
            )  # Green points for reprojected corners

        # Show the image
        cv2.imshow("Reprojected Points", img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


def draw_reprojected_points(images, objpoints, imgpoints, mtx, dist):
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        _, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)

        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(
            np.float32([[3, 0, 0], [0, 3, 0], [3, 2, 0]]), rvec, tvec, mtx, dist
        )

        # Convert points to tuples
        imgpoints_tuple = (int(imgpoints[i][0][0][0]), int(imgpoints[i][0][0][1]))
        imgpts_tuples = [(int(pt[0][0]), int(pt[0][1])) for pt in imgpts]

        # Draw axis
        img = cv2.line(img, imgpoints_tuple, imgpts_tuples[0], (255, 0, 0), 5)  # x-axis
        img = cv2.line(img, imgpoints_tuple, imgpts_tuples[1], (0, 255, 0), 5)  # y-axis
        img = cv2.line(img, imgpoints_tuple, imgpts_tuples[2], (0, 0, 255), 5)  # z-axis

        # Show the image
        cv2.imshow("Reprojected Points", img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    return rvec, tvec


# Define the chessboard size
chessboard_size = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp = objp * 20

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Load images
images = glob.glob("3D-Reconstruction/chessboard_images/*.png")

stored_image_names = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points
    if ret:

        stored_image_names.append(fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
np.savez(
    "3D-Reconstruction/calibration_data.npz",
    camera_matrix=mtx,
    dist_coeffs=dist,
    rvecs=rvecs,
    tvecs=tvecs,
)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Draw reprojected points to visually check the calibration
# draw_reprojected_points(stored_image_names, objpoints, imgpoints, mtx, dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

total_error = mean_error / len(objpoints)
print(f"Total reprojection error: {total_error}")
