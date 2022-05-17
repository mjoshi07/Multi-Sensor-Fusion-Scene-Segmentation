import argparse
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from FusionData import FusionDataset
from FusionNet import FusionNet, count_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint_path', type=str,
                        help='The path to the checkpoint to load.')
    parser.add_argument('-dp', '--dataset_path', type=str,
                        help='The path to the dataset.')
    parser.add_argument('-l', '--lidar', action='store_true',
                        help='Should lidar data be used for training. Default: False')
    parser.add_argument('-o', '--optical_flow', action='store_true',
                        help='Should flow data be used for training. Default: False')
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint_path
    dataset_path = args.dataset_path
    lidar = args.lidar
    optical_flow = args.optical_flow
    loss_fn = torch.nn.MSELoss()
    
    dataset = FusionDataset('../Data/vkitti', input_shape=(621, 187), in_mem=False)
    dataloader = DataLoader(dataset)
    
    model = FusionNet(out_channels=3, input_shape=(187, 621), lidar=lidar, optical_flow=optical_flow)
    params = count_parameters(model)
    print('===========================================================')
    print(f'Starting training with lidar: {lidar}, optical flow: {optical_flow}')
    print(f'Number of model parameters: {params}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print('CUDA enabled GPU found!')
        model = model.to(device)
    else:
        print('CUDA enabled GPU not found! Using CPU.')
    print('===========================================================')
    
    model.eval()
    
    with torch.no_grad():
        for i, (model_intput, gt) in enumerate(tqdm(dataloader)):
            model_intput = model_intput.float().cuda()
            gt = gt.float().cuda()
            model_output = model(model_intput)
            loss = loss_fn(model_output, gt)
            
            model_output = model_output.detach().cpu().numpy()
            model_output = model_output * 255
            gt = gt.detach().cpu().numpy()
            gt = gt * 255
            stacked = np.hstack((gt, model_output))
            
            cv2.imshow('', stacked)
            cv2.waitKey()
        
    