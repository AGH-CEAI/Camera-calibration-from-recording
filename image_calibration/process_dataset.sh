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
THRESHOLD=0.3
CHECKERBOARD_GRID="9 6"
# CHECKERBOARD_GRID="4 3"
CALIBRATION_SUBFOLDER_NAME=calibration

echo "[process_dataset] Processing directory: $1"

python3 ./image_calibration.py -i ${1}/${CALIBRATION_SUBFOLDER_NAME} --image-format ${IMAGE_FORMAT} -t ${THRESHOLD} --checkerboard-grid ${CHECKERBOARD_GRID}
python3 ./undistortion.py -i ${1} --image-format ${IMAGE_FORMAT} --crop

echo "[process_dataset] Done"
