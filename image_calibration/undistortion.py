#!/bin/python3

import argparse
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Calibration:
    mtx: np.array
    dist: np.array
    rvecs: np.array
    tvecs: np.array


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"> Loading calibration file from {args.config_file}")
    calibration = load_calibration(args.config_file)

    imgs_paths = get_imgs_paths(args.input_dir, args.img_format)
    img_shape = get_img_shape(imgs_paths[0])
    new_camera_mtx, roi = calculate_optimal_camera_matrix(img_shape, calibration)
    print(f"> Original image dims: {img_shape[1]}x{img_shape[0]} ({img_shape[2]} channels)")

    for cnt, path in enumerate(imgs_paths):
        print(f"> Processing image {cnt+1:02d}/{len(imgs_paths)}: {path.name}")
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Image '{path}' is None")

        undistorted_img = cv2.undistort(
            img, calibration.mtx, calibration.dist, None, new_camera_mtx
        )
        if args.crop:
            undistorted_img = crop_img_to_roi(undistorted_img, roi)
            undistorted_img = cv2.resize(undistorted_img, (img_shape[0], img_shape[1]))

        out_file = args.output_dir / path.name
        cv2.imwrite(str(out_file), undistorted_img)
        if args.verbose:
            print(f"> Saved undistorted image: {path.name}")

    print(f"> All undisroted files can be find in: {args.output_dir}")
    print("> Done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the folder with images to undistort",
        type=Path,
        required=True,
    )
    parser.add_argument("--image-format", help="Image format extension", default="jpg", type=str)
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for the undisorted images",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Path to the calibration file (.npz format)",
        type=Path,
        default=None,
    )
    parser.add_argument("--crop", help="Enable cropping to the ROI", action="store_true")
    parser.add_argument("-v", "--verbose", help="Shows more information", action="store_true")

    args = parser.parse_args()
    args.img_format = args.image_format

    if args.output_dir is None:
        args.output_dir = args.input_dir / "undistorted"

    if args.config_file is None:
        args.config_file = args.input_dir / "calibration.npz"

    return args


def load_calibration(path: Path) -> Calibration:

    data = np.load(path)
    return Calibration(mtx=data["mtx"], dist=data["dist"], rvecs=data["rvecs"], tvecs=data["tvecs"])


def get_imgs_paths(input_folder: Path, img_format: str) -> list[Path]:
    imgs_paths = [im for im in input_folder.iterdir() if im.suffix == f".{img_format}"]
    print(f"> |INPUT FOLDER| {input_folder}: {len(imgs_paths)} images")
    return imgs_paths


def get_img_shape(img_path: Path) -> tuple[int, int, int]:
    img = cv2.imread(str(img_path))
    return img.shape


def calculate_optimal_camera_matrix(
    img_shape: np.array, calibration: Calibration
) -> tuple[np.array, np.array]:
    shape = (img_shape[1] * 10, img_shape[0] * 10)
    return cv2.getOptimalNewCameraMatrix(calibration.mtx, calibration.dist, shape, 1, shape)


def crop_img_to_roi(img: cv2.typing.MatLike, roi: cv2.typing.Rect):
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


if __name__ == "__main__":
    main()
