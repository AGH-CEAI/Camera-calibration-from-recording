import numpy as np
import cv2

point_list = [
    [0, 0, 0],
    [0, 0, 89],
    [0, 180, 0],
    [0, 180, 89],
    [39, 180, 0],
    [39, 180, 89],
]

# TODO: setup this variables
mtx, dist = None, None


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, point_gathered
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        point_gathered = True


# def worldLine2imageLine(image,mtx,dist,rvec,tvec,worldLine):


if __name__ == "__main__":
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)

    capturer = cv2.VideoCapture("nagranie.dav")
    ret, frame = capturer.read()
    capturer.release()

    img_points = []

    for point in point_list:
        point_gathered = False
        while not point_gathered:
            frame_plus_txt = cv2.putText(
                frame.copy(),
                "Locate point " + str(point),
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("image", frame_plus_txt)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord("a"):
                print(mouseX, mouseY)
        img_points.append([mouseX, mouseY])
    cv2.destroyAllWindows()

    calibration_data = np.load("calibration.npz")
    for k in calibration_data.keys():
        exec(f"{k}=calibration_data['{k}']")

    ret, rvec, tvec = cv2.solvePnP(
        np.array(point_list, np.float32), np.array(img_points, np.float32), mtx, dist
    )
    R, _ = cv2.Rodrigues(rvec)
    print(-R.T @ tvec)

    np.savez("extrinsics.npz", rvec=rvec, tvec=tvec)

    # points_on_aline=np.array([[[0]*180,[i for i in range(180)],[0]*180]], np.float32)

    # points_2d, _ = cv2.projectPoints(points_on_aline,
    #                              rvec, tvec,
    #                              mtx,
    #                              dist)

    # for i in range(len(points_2d)):
    #     x,y=int(points_2d[i][0][0]),int(points_2d[i][0][1])
    #     cv2.circle(frame,(x,y),1,(255,0,0),-1)

    # cv2.imshow("image",frame)
    # cv2.waitKey(0)
