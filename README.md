# Camera calibration tools
This repository contains simple OpenCV scripts for performing the camera calibration.

## Camera calibration
- `./camera_calibration`
> Original scripts by Paweł Kolendo (@NieTrawisz) dedicated for **recordings** in the `*.dav` format.

## Image calibration
- `./image_calibration`
> Scripts for pre-processing caputered frames



### Calculate calibration params
```bash
./image_calibration/image_calibration.py -i "path/to/calibration/images/folder" -o "output/folder" --image-format "png" -t 0.2 --grid 9 6
```
Type `-h` for help.

### Apply calibration to images
```bash
./image_calibration/undistortion.py -i "path/to/images/folder" -c "path/to/calibration.npz" -o "output/folder" --image-format "png"
```
Type `-h` for help (for instance, there is an option to crop the image with the `--crop` flag)

### Process a dataset
Edit the `image_calibration/process_dataset.sh` file to your needs.
```bash
./image_calibration/process_dataset.sh "path/to/dataset/images/folder"
```
----

## Requirements
```
numpy
opencv-python
```
