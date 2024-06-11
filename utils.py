import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to project 2D point into 3D world coordinates when z=0
def project_to_world_ground(H, image_point):
    image_point_homog = np.append(image_point, 1)  # Convert to homogeneous coordinates
    world_point_homog = np.dot(H, image_point_homog)
    world_point_homog /= world_point_homog[-1]  # Normalize
    world_point = world_point_homog[:2]  # Discard the homogeneous coordinate
    return world_point


# Function to project a 2D image point to a 3D ray
def project_image_point_to_ray(mtx, R, tvec, image_point):
    image_point_homog = np.append(image_point, 1)  # Convert to homogeneous coordinates
    camera_point = np.dot(
        np.linalg.inv(mtx), image_point_homog
    )  # Convert to camera coordinates
    camera_point /= camera_point[-1]  # Normalize

    # Convert to world coordinates
    world_point = np.dot(
        R.T, (np.expand_dims(camera_point, axis=1) - tvec).reshape(3, 1)
    )
    world_point = world_point.flatten()

    return world_point


# Function to find the intersection of the ray with the plane
def find_intersection_with_plane(
    ray_origin,
    ray_direction,
    plane_point,
    plane_normal,
    tvec,
    mtx,
    R,
    image_point,
    camera_position_world,
):

    ray_origin = camera_position_world.flatten()
    ray_direction = project_image_point_to_ray(mtx, R, tvec, image_point) - ray_origin

    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)

    d = np.dot(plane_point - ray_origin, plane_normal) / np.dot(
        ray_direction, plane_normal
    )
    intersection = ray_origin + d * ray_direction

    return intersection


def plot_ray(
    R,
    camera_center_world,
    ray_direction_list,
    intersection_point_list,
    world_coord_A,
    world_coord_B,
    X=0,
    Y=0,
    Z=0,
    plane_point=None,
    plane_normal=None,
    plot_plane=True,
):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if plot_plane:
        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.5)

    intersection_points = np.array(intersection_point_list)

    # Plot camera position
    ax.scatter(
        camera_center_world[0],
        camera_center_world[1],
        camera_center_world[2],
        color="r",
        marker="o",
        label="Camera Position",
    )

    # Plot intersection point
    ax.scatter(
        intersection_points[:, 0],
        intersection_points[:, 1],
        intersection_points[:, 2],
        color="b",
        marker="o",
        label="Intersection Point",
    )

    # Plot intersection point
    ax.scatter(
        world_coord_A[0],
        world_coord_A[1],
        world_coord_A[2],
        color="b",
        marker="o",
        label="Intersection Point",
    )

    # Plot intersection point
    ax.scatter(
        world_coord_B[0],
        world_coord_B[1],
        world_coord_B[2],
        color="b",
        marker="o",
        label="Intersection Point",
    )

    # Plot ray from camera
    for ray_direction in ray_direction_list:
        scale = -camera_center_world[2] / ray_direction[2]
        ax.plot(
            [camera_center_world[0], camera_center_world[0] + scale * ray_direction[0]],
            [camera_center_world[1], camera_center_world[1] + scale * ray_direction[1]],
            [camera_center_world[2], camera_center_world[2] + scale * ray_direction[2]],
            color="g",
        )

    # Set axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rays from Camera")

    # Add legend
    ax.legend()

    # Show plot
    plt.show(block=False)
    plt.pause(10)
    plt.close()


def project_pose_to_world(keypoints_excluding_ankles, ankle_keypoints):

    # Load calibration data
    calibration_data = np.load("3D-Reconstruction/calibration_data.npz")
    mtx = calibration_data["camera_matrix"]
    dist = calibration_data["dist_coeffs"]
    rvecs = calibration_data["rvecs"]
    tvecs = calibration_data["tvecs"]

    import json

    with open("3D-Reconstruction/room_annotation/annotated_points.json", "r") as f:
        image_points = json.load(f)

    image_points = np.array(
        [point["coordinates"] for point in image_points], dtype="float32"
    )

    world_coords = np.array(
        [[0, 0, 0], [0, 100, 0], [100, 100, 0], [100, 0, 0]], dtype="float32"
    )

    # Estimate camera pose
    ret, rvec, tvec = cv2.solvePnP(world_coords, image_points, mtx, dist)

    # Plot camera orientation
    R, _ = cv2.Rodrigues(rvec)

    R_inv = R.T

    camera_position_world = -R_inv.dot(tvec)

    # Projection matrix
    # P = np.dot(mtx, np.hstack((R, tvec)))

    # The camera's forward direction vector in camera coordinates
    camera_forward = np.array([0, 0, 1])

    # Transform the camera forward direction to world coordinates
    camera_dir = R.T.dot(camera_forward) * 100  # scale by 100

    # Calculate homography
    H, _ = cv2.findHomography(image_points, world_coords)

    image_point = np.array(ankle_keypoints[0], dtype="float32")
    world_coord_A = project_to_world_ground(H, image_point)
    world_coord_A = np.append(world_coord_A, 0)

    image_point = np.array(ankle_keypoints[1], dtype="float32")
    world_coord_B = project_to_world_ground(H, image_point)
    world_coord_B = np.append(world_coord_B, 0)

    # Compute normal vector to the plane containing points A and B and as flat as possible along Z-axis
    AB = world_coord_B - world_coord_A
    k = np.array([0, 0, 1])
    n = np.cross(AB, k)
    n = n / np.linalg.norm(n)  # Normalize the vector

    # Define the plane using point A and normal vector n
    x0, y0, z0 = world_coord_A
    a, b, c = n

    x_range = np.linspace(0, 100, 100)
    z_range = np.linspace(0, 100, 100)
    x_grid, z_grid = np.meshgrid(x_range, z_range)

    # Calculate corresponding y values on the plane
    y_grid = (a * (x0 - x_grid) + c * (z0 - z_grid)) / -b + y0

    # Project the 2D image point to a 3D ray
    ray_origin = camera_position_world.flatten()

    ray_direction_list = []
    intersection_point_list = []
    # Compute the plane defined by the ankle keypoints
    for keypoint in keypoints_excluding_ankles:
        x, y = keypoint

        # Compute 3d rays and ground points
        ray_direction = project_image_point_to_ray(mtx, R, tvec, (x, y)) - ray_origin
        intersection_point = find_intersection_with_plane(
            ray_origin,
            ray_direction,
            world_coord_A,
            n,
            tvec,
            mtx,
            R,
            (x, y),
            camera_position_world,
        )
        ray_direction_list.append(ray_direction)
        intersection_point_list.append(intersection_point)

    # Plot ray and points
    # plot_ray(R, camera_position_world, ray_direction_list, intersection_point_list, world_coord_A, world_coord_B, x_grid, y_grid, z_grid)

    return (
        intersection_point_list,
        world_coord_A,
        world_coord_B,
        camera_position_world,
        camera_dir,
    )
