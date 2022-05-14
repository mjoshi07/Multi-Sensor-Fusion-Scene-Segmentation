"""
The following functions have been taken from the following repo
https://github.com/mjoshi07/Visual-Sensor-Fusion
"""

import struct
import numpy as np
import open3d as o3d


def convert_bin_to_pcd(binary_file, pcd_filepath):
    list_pcd = []
    size_float = 4
    with open(binary_file, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    o3d.io.write_point_cloud(pcd_filepath, pcd)


def convert_3d_to_homo(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def convert_3D_to_2D(R0, P, L2C, lidar_pts):
    '''
    Input: 3D points in Velodyne Frame [nx3]
    Output: 2D Pixels in Image Frame [nx2]
    '''
    R0_homo = np.vstack((R0, [0,0,0]))

    R0_homo_2 = np.hstack((R0_homo, [[0],[0],[0],[1]]))

    P_R0 = np.dot(P, R0_homo_2)

    P_R0_Rt = np.dot(P_R0, np.vstack((L2C, [0, 0, 0, 1])))

    pts_3d_homo = convert_3d_to_homo(lidar_pts)
    P_R0_Rt_X = np.dot(P_R0_Rt, pts_3d_homo.T)

    pts_2d_homo = P_R0_Rt_X.T

    pts_2d_homo /= pts_2d_homo[:, 2].reshape(-1, 1)
    pts_2d = pts_2d_homo[:, :2]


    return pts_2d


def remove_lidar_points_beyond_img(R0, P, L2C, lidar_pts, xmin, ymin, xmax, ymax, clip_distance=2.0):
    """ Filter lidar points, keep only those which lie inside image """
    pts_2d = convert_3D_to_2D(R0, P, L2C, lidar_pts)
    inside_pts_indices = ((pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax))

    # pc_velo are the points in LiDAR frame
    # therefore x-axis is in forward direction
    # we want to keep objects that are at least clip_distance away from sensor
    # X points are at 0th index column
    inside_pts_indices = inside_pts_indices & (lidar_pts[:, 0] > clip_distance)
    pts_3d_inside_img = lidar_pts[inside_pts_indices, :]

    return pts_3d_inside_img, pts_2d, inside_pts_indices


def get_lidar_on_image(R0, P, L2C, lidar_pts, size):
    """ Project LiDAR points to image """
    pts_3d_inside_img, all_pts_2d, fov_inds = remove_lidar_points_beyond_img(R0, P, L2C, lidar_pts, 0, 0,
                                                                             size[0], size[1], 2)

    return pts_3d_inside_img, all_pts_2d[fov_inds, :]
