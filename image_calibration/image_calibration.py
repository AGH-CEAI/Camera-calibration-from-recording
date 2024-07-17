#!/bin/python3

import argparse
import cv2
import numpy as np
from pathlib import Path


def main():
    args = parse_args()

    imgs_paths = get_calibration_imgs_paths(args.input_dir, args.img_format)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checkerboard_grid = args.checkerboard_grid

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_grid[0] * checkerboard_grid[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard_grid[0], 0 : checkerboard_grid[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    world_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane

    used_imgs_cnt = 0
    for cnt, path in enumerate(imgs_paths):
        print(f"> Processing image {cnt+1:02d}/{len(imgs_paths):02d}: {path.name}")
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Image '{path}' is None")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        checkboard_found, raw_corners = cv2.findChessboardCorners(img_gray, checkerboard_grid, None)

        if not checkboard_found:
            msg = "ERR: Missing points"
            print(f"> {msg}")
            cv2_put_text(img, msg)

        else:
            corners = cv2.cornerSubPix(img_gray, raw_corners, (11, 11), (-1, -1), criteria)

            if is_any_checkboard_corner_near_corner(
                checkerboard_grid, img_gray, corners, args.threshold
            ):
                world_points.append(objp)
                image_points.append(corners)
                cv2.drawChessboardCorners(img, checkerboard_grid, corners, checkboard_found)
                used_imgs_cnt += 1

            else:
                msg = "ERR: Checkerboard is too far from corners"
                print(f"> {msg}")
                cv2_put_text(img, msg)

        if args.verbose:
            cv2.imshow("Preview", img)
            cv2.waitKey(1000)

    print(
        f"> Finished image analysis. Measurements are taken from {used_imgs_cnt} out of {len(imgs_paths)} images."
    )
    if used_imgs_cnt == 0:
        raise ValueError(
            "None of the calibration images can be used! Try to change the threshold argument."
        )

    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_points, image_points, img_gray.shape[::-1], None, None
    )
    print("> Calculated camera parameters")

    error = calculate_reprojection_error(world_points, image_points, rvecs, tvecs, mtx, dist)
    print(f"{'':>8s}> Total re-projection error: {error/len(world_points):0.6f}")
    print(
        f"{'':>8s}> (TIP: The closer the re-projection error is to zero, the more accurate the parameters are)"
    )

    if args.verbose:
        print(
            f"> Calibration params: \n  mtx={mtx}\n  dist={dist}\n  rvecs={rvecs}\n  tvecs={tvecs}"
        )

    np.savez(
        args.output_file,
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    print(f"> Saved configuration to the {args.output_file} file")
    print("> Done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to the folder with calibration images",
        type=Path,
        required=True,
    )
    parser.add_argument("--image-format", help="Image format extension", default="jpg", type=str)
    parser.add_argument(
        "-o",
        "--output-file",
        help="Output path for the camera calibration file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Calibration point threshold value",
        type=float,
        required=True,
        default=0.2,
    )
    parser.add_argument("-v", "--verbose", help="Show calibration video", action="store_true")

    parser.add_argument(
        "--grid",
        "--checkerboard-grid",
        help="Checkerboard grid size (2 params: width height)",
        type=int,
        nargs="+",
        required=True,
    )

    args = parser.parse_args()
    args.checkerboard_grid = tuple(args.grid)
    args.img_format = args.image_format

    if args.output_file is None:
        args.output_file = args.input_dir.parent / "calibration.npz"
    else:
        args.output_file = args.output_file.with_suffix(".npz")

    return args


def get_calibration_imgs_paths(input_folder: Path, img_format: str) -> list[Path]:
    imgs_paths = [im for im in input_folder.iterdir() if im.suffix == f".{img_format}"]
    print(f"> |INPUT FOLDER| {input_folder}: {len(imgs_paths)} images")
    return imgs_paths


def cv2_put_text(img: cv2.typing.MatLike, text: str) -> None:
    cv2.putText(
        img,
        text,
        (32, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def is_any_checkboard_corner_near_corner(
    checkerboard_grid: tuple[int, int],
    img_gray: cv2.typing.MatLike,
    corners: cv2.typing.MatLike,
    calib_point_threshold: float,
) -> bool:
    # TODO: rewrite these indexes in more meaningful way
    threshold = calib_point_threshold
    return (
        is_in_threshold_range(corners[0, 0, :].reshape(2), img_gray.shape, threshold)
        or is_in_threshold_range(
            corners[checkerboard_grid[0] - 1, 0, :].reshape(2),
            img_gray.shape,
            threshold,
        )
        or is_in_threshold_range(
            corners[-checkerboard_grid[0], 0, :].reshape(2), img_gray.shape, threshold
        )
        or is_in_threshold_range(corners[-1, 0, :].reshape(2), img_gray.shape, threshold)
    )


def is_in_threshold_range(point: np.array, shape: np.array, calib_point_threshold: float) -> bool:
    # TODO: rewrite these indexes in more meaningful way
    dist = []
    corners = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    for corner in corners:
        dist.append(np.sqrt((point[0] - corner[0]) ** 2 + (point[1] - corner[1]) ** 2))

    threshold = np.sqrt(shape[0] ** 2 + shape[1] ** 2) * calib_point_threshold
    return any(d < threshold for d in dist)


def calculate_reprojection_error(
    world_points: list, img_points: list, rvecs, tvecs, mtx, dist
) -> float:
    mean_error = 0.0
    for i in range(len(world_points)):
        imgpoints2, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error


if __name__ == "__main__":
    main()
