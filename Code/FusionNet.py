import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np


class FusionNet(nn.Module):
    def __init__(self, out_channels, input_shape, lidar=False, optical_flow=False):
        super(FusionNet, self).__init__()
        self.input_shape = input_shape

        self.lidar = lidar
        self.optical_flow = optical_flow

        vgg_16_model = torchvision.models.vgg16(pretrained=True)
        return_layers = {'16': 'max_pool1', '23': 'max_pool2', '30': 'max_pool3'}
        self.vgg_16 = IntermediateLayerGetter(vgg_16_model.features, return_layers=return_layers)

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

        rgb_img = rgb_img.permute(0, 3, 1, 2)
        rgb_output = self.vgg_16(rgb_img)
        rgb_o1 = rgb_output['max_pool1']
        rgb_o2 = rgb_output['max_pool2']
        rgb_o3 = rgb_output['max_pool3']

        intermediate_sum_1 = rgb_o1
        intermediate_sum_2 = rgb_o2
        intermediate_sum_3 = rgb_o3

        if self.lidar:
            lidar_img = lidar_img.permute(0, 3, 1, 2)
            lidar_output = self.vgg_16(lidar_img)
            lidar_o1 = lidar_output['max_pool1']
            lidar_o2 = lidar_output['max_pool2']
            lidar_o3 = lidar_output['max_pool3']

            intermediate_sum_1 += lidar_o1
            intermediate_sum_2 += lidar_o2
            intermediate_sum_3 += lidar_o3

        if self.optical_flow:
            oflow_img = oflow_img.permute(0, 3, 1, 2)
            oflow_output = self.vgg_16(oflow_img)
            oflow_o1 = oflow_output['max_pool1']
            oflow_o2 = oflow_output['max_pool2']
            oflow_o3 = oflow_output['max_pool3']

            intermediate_sum_1 += oflow_o1
            intermediate_sum_2 += oflow_o2
            intermediate_sum_3 += oflow_o3

        output_3 = self.conv2d_1d_512(intermediate_sum_3)
        output_3 = self.up(output_3)

        output_2 = self.conv2d_1d_512(intermediate_sum_2)
        output_2 = nn.Upsample(size=output_3.size()[2:], mode="bilinear", align_corners=True)(output_2)
        output_2 = output_3 + output_2
        output_2 = self.up(output_2)

        output_1 = self.conv2d_1d_256(intermediate_sum_1)
        output_1 = nn.Upsample(size=output_2.size()[2:], mode="bilinear", align_corners=True)(output_1)
        output_1 = output_1 + output_2
        output = nn.Upsample(size=self.input_shape, mode="bilinear", align_corners=True)(output_1)

        return output


def split_input(input_data, lidar, optical_flow):

    x = input_data
    if lidar and optical_flow:
        return x[:, :, :, :3], x[:, :, :, 3:6], x[:, :, :, 6:]
    if not lidar and optical_flow:
        return x[:, :, :, :3], None, x[:, :, :, 3:]
    if lidar and not optical_flow:
        return x[:, :, :, :3], x[:, :, :, 3:], None

    return x[:, :, :, :3], None, None


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


if __name__ == "__main__":

    import FusionData as fd

    data_path = "../Data/Train"
    train_loader = fd.DataLoader(fd.FusionDataset(data_path, (187, 621)), batch_size=4, shuffle=True)
    train_iter = iter(train_loader)
    input_imgs, output_imgs = next(train_iter)
    print('input shape: ', input_imgs.shape)
    print('output shape: ', output_imgs.shape)

    model = FusionNet(out_channels=3, input_shape=(187, 621), lidar=True, optical_flow=True)
    print('number of trainable parameters =', count_parameters(model))

    import time
    from tqdm import tqdm

    iterations = 2
    t = time.time()
    for c in tqdm(range(iterations)):
        output = model(input_imgs.float())

    print("time taken for 1 iteration: ", (time.time() - t)/iterations)
    print('predicted shape: ', output.shape)
