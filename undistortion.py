import numpy as np
import cv2 as cv

recordings = ["nagranie.dav", "nagranie2.dav"]
# recordings=["short_test.avi"]

if __name__ == "__main__":
    for recording in recordings:
        # save undistorted video
        capturer = cv.VideoCapture("calibration recordings/" + recording)
        property_id = int(cv.CAP_PROP_FRAME_COUNT)
        video_length = int(cv.VideoCapture.get(capturer, property_id))
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
        calibration_data = np.load("calibration.npz")
        for k in calibration_data.keys():
            exec(f"{k}=calibration_data['{k}']")

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            mtx, dist, (ww, hh), 1, (ww, hh)
        )
        i = 0
        while capturer.isOpened():
            ret, frame = capturer.read()
            if ret:
                mapx, mapy = cv.initUndistortRectifyMap(
                    mtx, dist, None, newcameramtx, (ww, hh), 5
                )
                undistorted_frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
                # undistorted_frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                undistorted_frame = cv.rectangle(
                    undistorted_frame, (x, y), (x + w, y + h), (200, 100, 0), 5
                )
                # undistorted_frame = undistorted_frame[y : y + h, x : x + w]
                undistorted_frame = cv.resize(undistorted_frame, (ww, hh))
                out_video.write(undistorted_frame)
            else:
                break
            print(f"Frame {i}/{video_length}")
            i += 1
        out_video.release()
        capturer.release()
        cv.destroyAllWindows()
