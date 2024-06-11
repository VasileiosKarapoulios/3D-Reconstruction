import vtk
import numpy as np


# Function to create a stick figure
def create_stick_figure(keypoints):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Define joint connections for the stick figure
    joint_connections = [
        [0, 1],  # nose and left-eye
        [0, 2],  # nose and right-eye
        [1, 3],  # left-eye and left-ear
        [2, 4],  # right-eye and right-ear
        [5, 6],  # left-shoulder and right-shoulder
        [5, 7],  # left-shoulder and left-elbow
        [5, 11],  # left-shoulder and left-hip
        [6, 8],  # right-shoulder and right-elbow
        [6, 12],  # right-shoulder and right-hip
        [7, 9],  # left-elbow and left-wrist
        [8, 10],  # right-elbow and right-wrist
        [11, 12],  # left-hip and right-hip
        [11, 13],  # left-hip and left-knee
        [12, 14],  # right-hip and right-knee
        [13, 15],  # left-knee and left-ankle
        [14, 16],  # right-knee and right-ankle
    ]

    # Add vertices for the stick figure
    for point in keypoints:
        points.InsertNextPoint(point[0], point[1], point[2])

    # Define lines to connect the vertices
    for connection in joint_connections:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(connection[0])
        lines.InsertCellPoint(connection[1])

    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Create an actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


# Function to set up camera using calibration parameters
def setup_camera(camera, camera_position_world, camera_dir):
    # Set camera position
    camera.SetPosition(
        camera_position_world[0], camera_position_world[1], camera_position_world[2]
    )

    # Calculate focal point based on camera direction
    focal_point = [camera_position_world[i] + camera_dir[i] for i in range(3)]
    camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])


# Function to render the scene
def render_scene(
    keypoints,
    renderer,
    interactor,
    render_window,
    camera_dir,
    camera_position_world,
    frame_number,
):

    # Clear previous contents
    renderer.RemoveAllViewProps()

    stick_figure_actor = create_stick_figure(keypoints)
    renderer.AddActor(stick_figure_actor)

    angle = np.radians(10)  # Convert angle to radians
    rotation_axis = [0, 1, 0]  # Rotate around y-axis
    rotation = vtk.vtkTransform()
    rotation.RotateWXYZ(-angle * 180 / np.pi, 0, 1, 0)  # Rotate around y-axis
    camera_dir = rotation.TransformVector(camera_dir)

    # Zoom out by multiplying camera direction by a factor
    camera_position_world[1] -= 100

    # Tilt camera view upwards by 10 degrees
    tilt_angle = np.radians(10)  # Convert angle to radians
    tilt_axis = np.cross(
        rotation_axis, camera_dir
    )  # Calculate axis perpendicular to rotation and camera direction
    tilt_rotation = vtk.vtkTransform()
    tilt_rotation.RotateWXYZ(
        tilt_angle * 180 / np.pi, *tilt_axis
    )  # Rotate around perpendicular axis
    camera_position_world = tilt_rotation.TransformPoint(camera_position_world)

    # Set camera
    # Create a camera and set up with calibration parameters
    camera = vtk.vtkCamera()
    setup_camera(camera, camera_position_world, camera_dir)

    # Add camera to renderer
    renderer.SetActiveCamera(camera)

    # Render
    render_window.Render()
    interactor.Render()

    # Save the rendering as a frame
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"3D-Reconstruction/Rendering/frame_{frame_number}.png")
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()
