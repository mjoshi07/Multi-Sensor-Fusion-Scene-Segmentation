import torch
import torch.nn as nn
import cv2


class UniModal(nn.Module):
    def __init__(self, in_channels):
        super(UniModal, self).__init__()
        self.conv2d_64_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.conv2d_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_512_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2d_512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x):
        x = self.conv2d_64_1(x)
        x = self.conv2d_64_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2d_128_1(x)
        x = self.conv2d_128_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2d_256_1(x)
        x = self.conv2d_256_2(x)
        x = self.conv2d_256_2(x)
        x = self.relu(x)
        o1 = self.max_pool(x)
        x = self.conv2d_512_1(o1)
        x = self.conv2d_512_2(x)
        x = self.conv2d_512_2(x)
        x = self.relu(x)
        o2 = self.max_pool(x)
        x = self.conv2d_512_2(o2)
        x = self.conv2d_512_2(x)
        x = self.conv2d_512_2(x)
        x = self.relu(x)
        o3 = self.max_pool(x)

        return o1, o2, o3


class FusionNet(nn.Module):
    def __init__(self, out_channels, input_shape, lidar=False, optical_flow=False):
        super(FusionNet, self).__init__()
        self.input_shape = input_shape
        self.RGB_block = UniModal(3)
        self.lidar = lidar
        self.optical_flow = optical_flow

        if self.lidar:
            self.LiDAR_block = UniModal(1)

        if self.optical_flow:
            self.OFLOW_block = UniModal(3)

        self.conv2d_1d_512 = nn.Conv2d(512, out_channels, kernel_size=(1, 1), padding=1, bias=False)
        self.conv2d_1d_256 = nn.Conv2d(256, out_channels, kernel_size=(1, 1), padding=1, bias=False)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        x - H x W x C
        C = 3(RGB), 1(LIDAR[DEPTH]), N(OPTICAL FLOW)
        """
        rgb_img, lidar_img, oflow_img = split_input(x, self.lidar, self.optical_flow)

        rgb_img = rgb_img.permute(2, 0, 1)
        rgb_img = rgb_img.unsqueeze(0)
        rgb_o1, rgb_o2, rgb_o3 = self.RGB_block(rgb_img)

        intermediate_sum_1 = rgb_o1
        intermediate_sum_2 = rgb_o2
        intermediate_sum_3 = rgb_o3

        if self.lidar:
            lidar_img = lidar_img.permute(2, 0, 1)
            lidar_img = lidar_img.unsqueeze(0)
            lidar_o1, lidar_o2, lidar_o3 = self.LiDAR_block(lidar_img)
            intermediate_sum_1 += lidar_o1
            intermediate_sum_2 += lidar_o2
            intermediate_sum_3 += lidar_o3

        if self.optical_flow:
            oflow_img = oflow_img.permute(2, 0, 1)
            oflow_img = oflow_img.unsqueeze(0)
            oflow_o1, oflow_o2, oflow_o3 = self.OFLOW_block(oflow_img)
            intermediate_sum_1 += oflow_o1
            intermediate_sum_2 += oflow_o2
            intermediate_sum_3 += oflow_o3

        output_3 = self.conv2d_1d_512(intermediate_sum_3)
        output_3 = self.up(output_3)

        output_2 = self.conv2d_1d_512(intermediate_sum_2)
        output_2 = nn.Upsample(size=output_3.size()[2:], mode="bilinear", align_corners=True)(output_2)
        output_2 = output_3 + output_2
        # output_2 = add_tensors(output_2, output_3)
        output_2 = self.up(output_2)

        output_1 = self.conv2d_1d_256(intermediate_sum_1)
        output_1 = nn.Upsample(size=output_2.size()[2:], mode="bilinear", align_corners=True)(output_1)
        output_1 = output_1 + output_2
        # output_1 = add_tensors(output_1, output_2)
        # output = self.up8(output_1)
        output = nn.Upsample(size=self.input_shape, mode="bilinear", align_corners=True)(output_1)

        # output = resize_tensor(output, self.input_shape)

        return output


def resize_tensor(tnsor, target_shape):
    t_numpy = tnsor.permute(0, 2, 3, 1).detach().numpy()
    t_numpy = t_numpy.squeeze()
    t_numpy = cv2.resize(t_numpy, (target_shape[1], target_shape[0]))
    t_tensor = torch.from_numpy(t_numpy.astype(float))
    t_tensor = t_tensor.permute(2, 0, 1)
    t_tensor = t_tensor.unsqueeze(0)

    return t_tensor


def add_tensors(t1, t2):
    t1_size = (t1.size()[2], t1.size()[3])
    t2_tensor = resize_tensor(t2, t1_size)

    sum_t1_t2 = t1 + t2_tensor

    return sum_t1_t2


def split_input(input_data, lidar, optical_flow):

    x = torch.squeeze(input_data, 0)
    if lidar and optical_flow:
        return x[:, :, :3], x[:, :, 3:4], x[:, :, 4:]
    if lidar and not optical_flow:
        return x[:, :, :3], x[:, :, 3:], None
    if not lidar and optical_flow:
        return x[:, :, :3], None, x[:, :, 3:]

    return x, None, None


if __name__ == "__main__":

    import FusionData as fd

    data_path = "../Data/Train"
    train_loader = fd.DataLoader(fd.FusionDataset(data_path), batch_size=1, shuffle=True)
    train_iter = iter(train_loader)
    input_imgs, output_imgs = next(train_iter)
    print('input shape: ', input_imgs.shape)
    print('output shape: ', output_imgs.shape)

    model = FusionNet(out_channels=3, input_shape=(375, 1242), lidar=True, optical_flow=True)
    import time
    from tqdm import tqdm

    iterations = 5
    t = time.time()
    for c in tqdm(range(iterations)):
        output = model(input_imgs.float())

    print("time taken for 1 iteration: ", (time.time() - t)/iterations)
    print('predicted shape: ', output.shape)
