import numpy as np
import cv2 as cv

# recordings=["short_test2.avi"]
recordings = ["nagranie.dav", "nagranie2.dav"]
checkerborad_grid = (10, 7)
show_calibration_video = False
calibration_point_threshold = 0.2


def isInThresholdRange(point, shape):
    dist = []
    corners = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    for corner in corners:
        dist.append(np.sqrt((point[0] - corner[0]) ** 2 + (point[1] - corner[1]) ** 2))

    threshold = np.sqrt(shape[0] ** 2 + shape[1] ** 2) * calibration_point_threshold
    return (
        # point[1] >= shape[0] * (1 - calibration_point_threshold)
        # or point[1] < shape[0] * calibration_point_threshold
        # or point[0] >= shape[1] * (1 - calibration_point_threshold)
        # or point[0] < shape[1] * calibration_point_threshold
        # point[0] >= shape[1] * (1 - calibration_point_threshold)
        # or point[0] < shape[1] * calibration_point_threshold
        any(d < threshold for d in dist)
    )


if __name__ == "__main__":
    for recording in recordings:
        print("Processing: " + recording)
        capturer = cv.VideoCapture("calibration recordings/" + recording)
        property_id = int(cv.CAP_PROP_FRAME_COUNT)
        video_length = int(cv.VideoCapture.get(capturer, property_id))

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((checkerborad_grid[0] * checkerborad_grid[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : checkerborad_grid[0], 0 : checkerborad_grid[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        ww, hh, fps = (
            int(capturer.get(x))
            for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS,
            )
        )
        out_video = cv.VideoWriter(
            "calibration " + recording[:-4] + ".avi",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (ww, hh),
        )

        i = 0
        while capturer.isOpened():
            ret, frame = capturer.read()
            if frame is not None:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret2, corners = cv.findChessboardCorners(gray, checkerborad_grid, None)

                # If found, add object points, image points (after refining them)
                if ret2:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    if (
                        isInThresholdRange(corners2[0, 0, :].reshape(2), gray.shape)
                        or isInThresholdRange(
                            corners2[checkerborad_grid[0] - 1, 0, :].reshape(2),
                            gray.shape,
                        )
                        or isInThresholdRange(
                            corners2[-checkerborad_grid[0], 0, :].reshape(2), gray.shape
                        )
                        or isInThresholdRange(corners2[-1, 0, :].reshape(2), gray.shape)
                    ):
                        objpoints.append(objp)
                        imgpoints.append(corners2)

                        # Draw and display the corners
                        cv.drawChessboardCorners(frame, checkerborad_grid, corners2, ret)

                    else:
                        cv.putText(
                            frame,
                            "Checkerboard is too far from corners",
                            (100, 100),
                            cv.FONT_HERSHEY_SIMPLEX,
                            3,
                            (0, 0, 255),
                            2,
                            cv.LINE_AA,
                        )

                else:
                    cv.putText(
                        frame,
                        "Not all of checkerboard points are detected",
                        (100, 100),
                        cv.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )

                if show_calibration_video:
                    cv.imshow(
                        "mask",
                        cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
                    )
                    cv.waitKey(1)

                out_video.write(frame)

            if not ret:
                break

            print(f"Frame {i}/{video_length}")
            i += 1

            keyboard = cv.waitKey(30)
            if keyboard == "q" or keyboard == 27:
                break

        capturer.release()
        cv.destroyAllWindows()

    print("Calibrating...")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    np.savez(
        "calibration",
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
    )

    shape = gray.shape
    shape = (shape[1] * 10, shape[0] * 10)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, shape, 1, shape)

    print("Done!")

    for recording in recordings:
        # save undistorted video
        capturer = cv.VideoCapture("calibration recordings/" + recording)
        ww, hh, fps = (
            int(capturer.get(x))
            for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS,
            )
        )
        out_video = cv.VideoWriter(
            "undistorted " + recording[:-4] + ".avi",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (ww, hh),
        )
        i = 0
        while capturer.isOpened():
            ret, frame = capturer.read()
            if ret:
                undistorted_frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                undistorted_frame = undistorted_frame[y : y + h, x : x + w]
                undistorted_frame = cv.resize(undistorted_frame, (ww, hh))
                out_video.write(undistorted_frame)
            else:
                break
            print(f"Frame {i}/{video_length}")
            i += 1
        out_video.release()
        capturer.release()
        cv.destroyAllWindows()
