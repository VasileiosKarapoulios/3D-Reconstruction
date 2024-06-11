# 3D Reconstruction from Monocular setup

## Contents

1. Room calibration
2. Human bounding box detection & pose estimation
3. 2D points (body joints) to 3D points
4. 3D pose rendering (visualization)

#### Details

- Calibration data are not included in the repo, but consist of chessboard images around the room
- Human detector + Pose estimator model is pretrained yolov8
- 2D points are projected to 3D through estimating an X/Z plane using the feet points (ground) and finding the intersection of every other joint with that plane


## Example rendering:

https://github.com/VasileiosKarapoulios/3D-Reconstruction/assets/54540739/b03b8ca2-63ba-4b16-ac17-87bef066fec0

