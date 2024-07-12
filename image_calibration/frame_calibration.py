import argparse
import cv2
import numpy as np
from pathlib import Path

# recordings=["short_test2.avi"]
recordings = ["nagranie.dav", "nagranie2.dav"]  # -> args.input_folder
checkerborad_grid = (10, 7)  # -> args.checkerboard_grid
show_calibration_video = False  # -> args.verbose
calibration_point_threshold = 0.2  # -> args.threshold


def main():
    args = parse_args()

    imgs_paths = get_calibration_imgs_paths(args.input_folder, args.img_format)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checkerboard_grid = args.checkerboard_grid

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerborad_grid[0] * checkerborad_grid[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : checkerborad_grid[0], 0 : checkerborad_grid[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    world_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane

    for cnt, path in enumerate(imgs_paths):
        print(f"> Processing image {cnt}/{len(imgs_paths)}: {path}")
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Image '{path}' is None")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        checkboard_found, raw_corners = cv2.findChessboardCorners(img_gray, checkerboard_grid, None)

        if not checkboard_found:
            msg = "Not all of the checkerboard points are detected"
            print(f"> {msg}")
            cv2_put_text(img, msg)

        else:

            corners = cv2.cornerSubPix(img_gray, raw_corners, (11, 11), (-1, -1), criteria)

            if (
                is_in_threshold_range(corners[0, 0, :].reshape(2), img_gray.shape)
                or is_in_threshold_range(
                    corners[checkerboard_grid[0] - 1, 0, :].reshape(2),
                    img_gray.shape,
                )
                or is_in_threshold_range(
                    corners[-checkerboard_grid[0], 0, :].reshape(2), img_gray.shape
                )
                or is_in_threshold_range(corners[-1, 0, :].reshape(2), img_gray.shape)
            ):
                world_points.append(objp)
                image_points.append(corners)
                cv2.drawChessboardCorners(img, checkerboard_grid, corners, checkboard_found)

            else:
                msg = "Checkerboard is too far from corners"
                print(f"> {msg}")
                cv2_put_text(img, msg)

        if args.verbose:
            cv2.imshow("Preview", img)
            cv2.waitKey(1000)

    print("> Finished image analysis.")

    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_points, image_points, img_gray.shape[::-1], None, None
    )
    print(f"> Calibration mtx={mtx} dist={dist} rvecs={rvecs} tvecs={tvecs}")

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
        "--input-folder",
        help="Path to the folder with calibration images",
        type=Path,
        required=True,
    )
    parser.add_argument("--image-format", help="Image format extension", default="jpg", type=str)
    parser.add_argument(
        "-o",
        "--output-file",
        help="Output directory for the camera calibration",
        type=Path,
        default="./calibration.yaml",
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
    args.checkerboard_grid = tuple(args.checkerboard_grid)

    return args


def get_calibration_imgs_paths(input_folder: Path, img_format: str) -> list[Path]:
    imgs_paths = [im for im in input_folder.iterdir() if im.suffix == f".{img_format}"]
    print(f"> |INPUT FOLDER| {input_folder}: {len(imgs_paths)} images")
    return imgs_paths


def cv2_put_text(img: cv2.typing.MatLike, text: str) -> None:
    cv2.putText(
        img,
        text,
        (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def is_in_threshold_range(point: np.array, shape: np.array) -> bool:
    dist = []
    corners = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    for corner in corners:
        dist.append(np.sqrt((point[0] - corner[0]) ** 2 + (point[1] - corner[1]) ** 2))

    threshold = np.sqrt(shape[0] ** 2 + shape[1] ** 2) * calibration_point_threshold
    return any(d < threshold for d in dist)


if __name__ == "__main__":
    main()
