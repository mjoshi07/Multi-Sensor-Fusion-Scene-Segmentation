import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class FusionDataset(Dataset):
    def __init__(self, path, input_shape, in_mem=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.rgb_img_dir = os.path.join(path, "vkitti_1.3.1_rgb")
        self.lidar_img_dir = os.path.join(path, "vkitti_1.3.1_depthgt")
        self.oflow_img_dir = os.path.join(path, "vkitti_1.3.1_flowgt")
        self.seg_mask_dir = os.path.join(path, "vkitti_1.3.1_scenegt")
        self.in_mem = in_mem
        self.input_shape = input_shape
        self.mean = mean
        self.std = std

        if not (os.path.exists(self.rgb_img_dir) and os.path.exists(self.lidar_img_dir)\
                and os.path.exists(self.oflow_img_dir) and os.path.exists(self.seg_mask_dir)):
            raise ValueError(f'The path {path} does not have required directory structure!')

        if self.in_mem:
            self.rgb_imgs, self.lidar_imgs, self.oflow_imgs, self.seg_imgs = self._load_images()
        else:
            self.image_paths = self._load_images()

    def __len__(self):
        if self.in_mem:
            return len(self.rgb_imgs)
        else:
            return len(self.image_paths)

    def _load_images(self):
        basename = os.path.basename(self.oflow_img_dir)
        dir_0001 = "0001"
        # dir_0002 = "0002"
        # dir_0006 = "0006"
        # dir_0018 = "0018"
        # dir_0020 = "0020"

        dirs = [dir_0001]

        end_img_paths = []
        for data_dir in dirs:
            for root, _, files in os.walk(os.path.join(self.oflow_img_dir, data_dir)):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    useful_path = full_path.split(basename)[1]
                    end_img_paths.append(useful_path[1:])

        if self.in_mem:
            rgb_images = []
            lidar_images = []
            oflow_images = []
            seg_images = []

            for img_name in end_img_paths:
                rgb_img = cv2.imread(os.path.join(self.rgb_img_dir, img_name))
                rgb_img = cv2.resize(rgb_img, self.input_shape)
                # lidar_img = cv2.imread(os.path.join(self.lidar_img_dir, img_name), 0)
                lidar_img = self.read_lidar_vkitti(img_name)
                lidar_img = cv2.resize(lidar_img, self.input_shape)
                oflow_img = self.read_oflow_vkitti(img_name)
                oflow_img = cv2.resize(oflow_img, self.input_shape)
                # oflow_img = cv2.imread(os.path.join(self.oflow_img_dir, img_name))
                seg_img = cv2.imread(os.path.join(self.seg_mask_dir, img_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                seg_img = cv2.resize(seg_img, self.input_shape)

                rgb_images.append(rgb_img)
                lidar_images.append(lidar_img)
                oflow_images.append(oflow_img)
                seg_images.append(seg_img)

            return rgb_images, lidar_images, oflow_images, seg_images
        else:
            return end_img_paths

    def __getitem__(self, idx):
        if self.in_mem:
            rgb_img = self.rgb_imgs[idx]
            lidar_img = self.lidar_imgs[idx]
            oflow_img = self.oflow_imgs[idx]
            seg_img = self.seg_imgs[idx]
        else:
            img_name = self.image_paths[idx]
            rgb_img = cv2.imread(os.path.join(self.rgb_img_dir, img_name))
            rgb_img = cv2.resize(rgb_img, self.input_shape)
            # lidar_img = cv2.imread(os.path.join(self.lidar_img_dir, img_name), 0)
            lidar_img = self.read_lidar_vkitti(img_name)
            lidar_img = cv2.resize(lidar_img, self.input_shape)
            oflow_img = self.read_oflow_vkitti(img_name)
            oflow_img = cv2.resize(oflow_img, self.input_shape)
            # oflow_img = cv2.imread(os.path.join(self.oflow_img_dir, img_name))
            seg_img = cv2.imread(os.path.join(self.seg_mask_dir, img_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            seg_img = cv2.resize(seg_img, self.input_shape)

        rgb_img = self.normalize(rgb_img)
        lidar_img = self.normalize(lidar_img)
        oflow_img = self.normalize(oflow_img)
        stacked = np.dstack((rgb_img, lidar_img, oflow_img))
        input_img = torch.from_numpy(stacked)

        seg_img = self.normalize(seg_img)
        seg_img = torch.from_numpy(seg_img)
        seg_img = seg_img.permute(2, 0, 1)

        return input_img, seg_img

    def read_oflow_vkitti(self, img_name):
        """
        https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/
        Change pixel range to (0, 255)
        """
        bgr = cv2.imread(os.path.join(self.oflow_img_dir, img_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w, c = bgr.shape
        assert bgr.dtype == np.uint16 and c == 3
        invalid = bgr[..., 0] = 0
        out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2: 0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        out_flow[invalid] = 0
        out_flow = out_flow - np.min(out_flow)
        out_flow = np.float32(out_flow * 255.0 / np.max(out_flow))
        zeros = np.zeros(out_flow.shape[:2], dtype=np.float32)
        out_flow = np.dstack((out_flow, zeros))
        return out_flow

    def read_lidar_vkitti(self, img_name):
        """
        pixel value = 1 represents distance of 1 cm in real world
        Change pixel range to (0, 255)
        """
        bgr = cv2.imread(os.path.join(self.lidar_img_dir, img_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bgr = np.float32(bgr * 255.0 / np.max(bgr))
        bgr = np.dstack((bgr, bgr, bgr))
        return bgr

    def normalize(self, img):

        img = img.astype(float)
        img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
        # img[:, :, 0] = (img[:, :, 0] - self.mean[0]) / (self.std[0])
        # img[:, :, 1] = (img[:, :, 1] - self.mean[1]) / (self.std[1])
        # img[:, :, 2] = (img[:, :, 2] - self.mean[2]) / (self.std[2])
        img[:, :, 0] = (img[:, :, 0] - np.mean(img[:, :, 0])) / np.std(img[:, :, 0])
        img[:, :, 1] = (img[:, :, 1] - np.mean(img[:, :, 1])) / np.std(img[:, :, 1])
        img[:, :, 2] = (img[:, :, 2] - np.mean(img[:, :, 2])) / np.std(img[:, :, 2])

        return img

if __name__ == "__main__":

    data_path = "../Data/Train"
    train_loader = DataLoader(FusionDataset(data_path, (187, 621), in_mem=True), batch_size=1, shuffle=True)
    train_iter = iter(train_loader)
    input_imgs, output_imgs = next(train_iter)
    print(input_imgs.shape, output_imgs.shape)
