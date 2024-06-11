import torch
import cv2
import numpy as np

from utils import project_pose_to_world
from ultralytics import YOLO

import vtk
from visualization import render_scene

# Create a VTK renderer and render window
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Set window size to 1080p
render_window.SetSize(1920, 1080)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a model
model = YOLO("yolov8l-pose.pt")
model.to(device)

# Open the video capture
cap = cv2.VideoCapture(0)

# Set the resolution to 1080p (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the video capture is opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_counter = 0
ankle_indices = [15, 16]

try:
    while True:

        world_points = []

        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame_counter == 250:
            break

        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            for box_conf, keypointxy in zip(boxes.conf, keypoints.xy):
                if box_conf < 0.5:
                    continue

                # Create a boolean mask with True values
                mask = np.ones(keypointxy.shape[0], dtype=bool)

                # Set the positions of ankle_indices to False
                mask[ankle_indices] = False

                # Select keypoints using the mask
                keypoints_excluding_ankles = keypointxy[mask]

                # Select only ankle keypoints
                ankle_keypoints = keypointxy[ankle_indices]
                (
                    intersection_point_list,
                    world_coord_A,
                    world_coord_B,
                    camera_position_world,
                    camera_dir,
                ) = project_pose_to_world(
                    keypoints_excluding_ankles.cpu(), ankle_keypoints.cpu()
                )
                world_points.append(
                    intersection_point_list + [world_coord_A] + [world_coord_B]
                )

            # result.save(filename=f"yolo_predictions/frame_{frame_counter}.png")

        # Render the scene
        render_scene(
            world_points[0],
            renderer,
            interactor,
            render_window,
            camera_dir,
            camera_position_world,
            frame_counter,
        )

        frame_counter += 1

finally:
    # When everything's done, release the capture
    cap.release()
    cv2.destroyAllWindows()
