# Camera calibration tools
This repository contains simple OpenCV scripts for performing the camera calibration.

## Camera calibration
- `./camera_calibration`
Original scripts by Paweł Kolendo (@NieTrawisz) dedicated for **recordings** in the `*.dav` format.

## Image calibration
- `./image_calibration`
Scripts for pre-processing caputered frames

### Calculate camera calibration
```bash
./image_calibration/frame_calibration.py -i "/path/to/calibration/images/folder" --image-format "png" -t 0.2 --grid 9 6
```
Type `-h` for help.

### Apply calibration to images