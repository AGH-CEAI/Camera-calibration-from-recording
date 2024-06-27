import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import cv2

point_list = [
    [0, 0, 0],
    [0, 0, 89],
    [0, 180, 0],
    [0, 180, 89],
    [39, 180, 0],
    [39, 180, 89],
]


if __name__ == "__main__":
    calibration_data = np.load("calibration.npz")
    for k in calibration_data.keys():
        exec(f"{k}=calibration_data['{k}']")

    extrinsics = np.load("extrinsics.npz")
    for k in extrinsics.keys():
        exec(f"{k}=extrinsics['{k}']")

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # defining all 3 axis
    x = [i[0] for i in point_list]
    y = [i[1] for i in point_list]
    z = [i[2] for i in point_list]

    # plotting
    ax.scatter3D(x, y, z, "green")
    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -R.T @ tvec
    camera_points = np.array(
        [
            [((-1) ** i) * 10, ((-1) ** (i // 2)) * 10, ((-1) ** (i // 4)) * 20]
            for i in range(8)
        ],
        np.float32,
    ).T
    world_camera_points = R.T @ (camera_points - tvec)
    for point1 in range(8):
        for point2 in range(8):
            ax.plot3D((world_camera_points[0,point1],world_camera_points[0,point2]),(world_camera_points[1,point1],world_camera_points[1,point2]),(world_camera_points[2,point1],world_camera_points[2,point2]),"red")
    # ax.plot_surface(
    #     world_camera_points[0, :],
    #     world_camera_points[1, :],
    #     world_camera_points[2, :],
    #     cmap="viridis",
    #     edgecolor="red",
    # )
    ax.set_title("3D line plot geeks for geeks")
    plt.show()
