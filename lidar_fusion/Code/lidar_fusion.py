import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import Calibration as ca
import Utils as ut


def fuse_with_lidar(data_path, out_dir=None, target_shape=(1242, 375),rgb_seg=True, save_video=True, display_video=True):

    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    rgb_img_dir = os.path.join(data_path, "rgb_images")
    seg_img_dir = os.path.join(data_path, "rgb_seg")
    flow_img_dir = os.path.join(data_path, "flow_seg")
    lidar_dir = os.path.join(data_path, "lidar_pcd")
    calib_file = os.path.join(data_path, os.path.join("calibration", "calib.txt"))

    calib = ca.CalibrationData(calib_file)
    R0 = calib.R0
    P = calib.P
    L2C = calib.L2C

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    img = cv2.imread(os.path.join(rgb_img_dir, os.listdir(rgb_img_dir)[0]))
    h, w = img.shape[:2]
    w_ = w
    h_ = h * 3
    fps = 10

    if save_video:
        writer = cv2.VideoWriter(os.path.join(out_dir, "fusion_video.mp4"), fourcc, fps, (w_, h_))

    img_idx = 0
    for root, dirs, files in os.walk(rgb_img_dir):
        for filename in files:
            if img_idx < 153:
                rgb_img = cv2.imread(os.path.join(rgb_img_dir, filename))
                if rgb_seg:
                    seg_img = cv2.resize(cv2.imread(os.path.join(seg_img_dir, filename)), target_shape)
                else:
                    seg_img = cv2.resize(cv2.imread(os.path.join(flow_img_dir, filename)), target_shape)

                img_idx += 1
                fused_img = rgb_img.copy()
                point_cloud = np.asarray(o3d.io.read_point_cloud(os.path.join(lidar_dir, filename[:-3] + "pcd")).points)
                pts_3D, pts_2D = ut.get_lidar_on_image(R0, P, L2C, point_cloud, (rgb_img.shape[1], rgb_img.shape[0]))

                for i in range(pts_2D.shape[0]):
                    depth = pts_3D[i, 0]

                    x = np.int32(pts_2D[i, 0])
                    y = np.int32(pts_2D[i, 1])

                    classID = np.float64(seg_img[y, x])  # classID is the unique RGB value of each class

                    # 510 has been calculated according to the clip distance, to get color value in range (0, 255)
                    # color = cmap[int(510.0 / depth), :]
                    pt = (x, y)
                    cv2.circle(fused_img, pt, 2, color=tuple(classID), thickness=-1)

                stacked_img = np.vstack((rgb_img, seg_img, fused_img))
                if save_video:
                    writer.write(stacked_img)

                if display_video:
                    cv2.namedWindow('stacked_img', cv2.WINDOW_KEEPRATIO)
                    cv2.imshow('stacked_img', stacked_img)
                    k = cv2.waitKey(1)
                    if k == ord('q') or k == 27:
                        exit()
                    if k == ord('p'):
                        cv2.waitKey(0)

    if save_video:
        writer.release()


if __name__ == "__main__":
    data_path = "../Data/KITTI_data"
    out_dir = "../Data/output"
    fuse_with_lidar(data_path, out_dir, target_shape=(1242, 375),rgb_seg=True, save_video=False, display_video=True)