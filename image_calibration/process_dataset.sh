#!/bin/bash

# The structure of the dataset directory:
# dataset/
#    calibration/
#       <calibration images>
#    undistorted/
#        <undistored images>
#    calibration.npz
#    <raw images>


# Pass the dataset directory as an argument
IMAGE_FORMAT=png
THRESHOLD=0.2
CHECKERBOARD_GRID="9 6"
CALIBRATION_SUBFOLDER_NAME=calibration

echo "[process_dataset] Processing directory: $1"

python3 ./frame_calibration.py -i ${1} --image-format ${IMAGE_FORMAT} -t ${THRESHOLD} --checkerboard-grid ${CHECKERBOARD_GRID}
python3 ./undistortion.py -i ${1} --image-format ${IMAGE_FORMAT}

echo "[process_dataset] Done"
