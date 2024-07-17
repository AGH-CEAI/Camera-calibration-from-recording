import numpy as np
import cv2

# TODO: fill these variables
mtx = None
dist = None
rvec = None
tvec = None


def worldLine2imageLine(image, mtx, dist, rvec, tvec, worldLine):
    points_on_aline = np.array(
        [
            [
                [worldLine[0]] * worldLine[1],
                [i for i in range(worldLine[1])],
                [worldLine[2]] * worldLine[1],
            ]
        ],
        np.float32,
    ).T

    image_copy = image.copy()
    points_2d, _ = cv2.projectPoints(points_on_aline, rvec, tvec, mtx, dist)

    for i in range(len(points_2d)):
        x, y = int(points_2d[i][0][0]), int(points_2d[i][0][1])
        cv2.circle(image_copy, (x, y), 3, (255, 0, 0), -1)

    return image_copy


def worldLine2imageLineUndistorted(image, mtx, dist, optimalMtx, roi, rvec, tvec, worldLine):
    points_on_aline = np.array(
        [
            [
                [worldLine[0]] * 2,
                [0, worldLine[1]],
                [worldLine[2]] * 2,
            ]
        ],
        np.float32,
    ).T

    image_undistorted = cv2.undistort(image, mtx, dist, None, optimalMtx)
    xx, yy, w, h = roi
    image_undistorted = image_undistorted[yy : yy + h, xx : xx + w]
    points_2d, _ = cv2.projectPoints(points_on_aline, rvec, tvec, mtx, dist)
    points_2d = cv2.undistortPoints(points_2d, mtx, dist, None, optimalMtx)

    image_undistorted = cv2.line(
        image_undistorted,
        (int(points_2d[0][0][0] - xx), int(points_2d[0][0][1] - yy)),
        (int(points_2d[1][0][0] - xx), int(points_2d[1][0][1] - yy)),
        (255, 0, 0),
        3,
    )

    # for i in range(len(points_2d)):
    #     x, y = int(points_2d[i][0][0]-xx), int(points_2d[i][0][1]-yy)
    #     cv2.circle(image_undistorted, (x, y), 3, (255, 0, 0), -1)

    return image_undistorted


if __name__ == "__main__":
    capturer = cv2.VideoCapture("nagranie.dav")
    ww, hh, fps = (
        int(capturer.get(x))
        for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        )
    )
    ret, frame = capturer.read()
    capturer.release()

    calibration_data = np.load("calibration.npz")
    for k in calibration_data.keys():
        exec(f"{k}=calibration_data['{k}']")

    extrinsics = np.load("extrinsics.npz")
    for k in extrinsics.keys():
        exec(f"{k}=extrinsics['{k}']")

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (10 * ww, 10 * hh), 1, (10 * ww, 10 * hh)
    )

    # points_on_aline=np.array([[[0]*180,[i for i in range(180)],[0]*180]], np.float32).T

    # points_2d, _ = cv2.projectPoints(points_on_aline,
    #                              rvec, tvec,
    #                              mtx,
    #                              dist)

    # for i in range(len(points_2d)):
    #     x,y=int(points_2d[i][0][0]),int(points_2d[i][0][1])
    #     cv2.circle(frame,(x,y),3,(255,0,0),-1)

    # img = worldLine2imageLine(frame, mtx, dist, rvec, tvec, [0, 180, 0])
    img = worldLine2imageLineUndistorted(
        frame, mtx, dist, newcameramtx, roi, rvec, tvec, [0, 180, 89]
    )

    try:
        while True:
            cv2.imshow("image", img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
