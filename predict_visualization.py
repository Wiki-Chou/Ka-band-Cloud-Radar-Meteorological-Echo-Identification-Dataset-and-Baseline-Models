# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Weijie Zou

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet import UNet, UNet2Plus, UNet3Plus, TransUNet, UNetWithSE
from utils.dataset import BasicDataset
from utils.eval import calculate_iou
from utils.data_vis import plot_img_and_mask, plot_imgs, plot_mask, Visualize_type


def predict_img(unet_type, net, full_img, device, scale_factor=1, out_threshold=0.5):
    """
    Generate a predicted mask for the given image using a U-Net model.
    Args:
    - unet_type (str): Type of U-Net model.
    - net (torch.nn.Module): The trained U-Net model.
    - full_img (np.ndarray): Input image (H, W, C).
    - device (torch.device): Computational device (CPU or GPU).
    - scale_factor (float): Scale factor for the input image.
    - out_threshold (float): Minimum probability threshold for the mask.
    
    Returns:
    - np.ndarray: Binary mask after thresholding.
    """
    net.eval()  # Set model to evaluation mode
    img = np.array(full_img, dtype=np.float32)

    # Handle NaN and inf values
    img[np.isnan(img)] = -999
    img[np.isinf(img)] = -999

    # Preprocess the image
    img = torch.from_numpy(BasicDataset.preprocess(unet_type, img, scale_factor))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output) if net.n_classes == 1 else F.softmax(output, dim=1)

        # Convert prediction to PIL image and resize to original size
        probs = probs.squeeze(0)
        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.shape[1]), transforms.ToTensor()])
        probs = tf(probs.cpu())

        # Return thresholded mask
        return probs.squeeze().cpu().numpy() > out_threshold


def slide_window_predict(unet_type, net, full_img, device, scale_factor=1, out_threshold=0.9):
    """
    Predict mask for large images using sliding window approach. Divides the image into smaller blocks.
    Args:
    - unet_type (str): Type of U-Net model.
    - net (torch.nn.Module): Trained U-Net model.
    - full_img (np.ndarray): Input image (H, W, C).
    - device (torch.device): Computational device (CPU or GPU).
    - scale_factor (float): Scale factor for the input image.
    - out_threshold (float): Minimum probability threshold for the mask.
    
    Returns:
    - np.ndarray: Binary mask for the entire image.
    """
    windows_size = 256
    stride = 100
    num_w = (full_img.shape[0] - windows_size) // stride + 1
    num_h = 1  # For simplicity, using 1 horizontal window, adjust as needed

    count_predict = np.zeros((full_img.shape[0], full_img.shape[1]))
    count_overlap = np.zeros((full_img.shape[0], full_img.shape[1]))

    for i in range(num_w):
        for j in range(num_h):
            img = full_img[i * stride:i * stride + windows_size, j * stride:j * stride + windows_size, :]
            mask = predict_img(unet_type, net, img, device, scale_factor, out_threshold)

            count_predict[i * stride:i * stride + windows_size, j * stride:j * stride + windows_size] += mask
            count_overlap[i * stride:i * stride + windows_size, j * stride:j * stride + windows_size] += 1

            # Edge handling for the last window
            if i == num_w - 1:
                img = full_img[full_img.shape[0] - windows_size:, j * stride:j * stride + windows_size, :]
                mask = predict_img(unet_type, net, img, device, scale_factor, out_threshold)
                count_predict[full_img.shape[0] - windows_size:, j * stride:j * stride + windows_size] += mask
                count_overlap[full_img.shape[0] - windows_size:, j * stride:j * stride + windows_size] += 1

            if j == num_h - 1 and i == num_w - 1:
                img = full_img[full_img.shape[0] - windows_size:, full_img.shape[1] - windows_size:, :]
                mask = predict_img(unet_type, net, img, device, scale_factor, out_threshold)
                count_predict[full_img.shape[0] - windows_size:, full_img.shape[1] - windows_size:] += mask
                count_overlap[full_img.shape[0] - windows_size:, full_img.shape[1] - windows_size:] += 1

    # Combine predictions
    mask = count_predict / count_overlap
    mask = mask > out_threshold

    # Post-process the mask
    mask[:, 150:][full_img[:, 150:, 0] >= -50] = True
    mask[:, 150:][full_img[:, 150:, 1] >= -15] = True
    mask[:, 150:][full_img[:, 150:, 2] >= 0] = True

    return mask


def get_args():
    """
    Parse command-line arguments.
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--unet_type', '-u', default='se', help='U-Net model type (v1/v2/v3/trans/se)')
    parser.add_argument('--input', '-i', default=[r'Data_test\imgs'], nargs='+', help='Input image file paths')
    parser.add_argument('--mask', '-m', default=[r'Data_test\masks'], nargs='+', help='Input mask file paths')
    parser.add_argument('--output', '-o', default=[r'output'], nargs='+', help='Output directories for results')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for mask prediction')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Scale factor for input images')
    return parser.parse_args()


def predict_met_echo(in_files, fn, output_dir, unet_type, net, scale_factor, out_threshold, device, if_true_mask=False, if_vis=False):
    """
    Predict echo mask from radar data (loaded from .npy files) using the trained U-Net model.
    Args:
    - in_files (str): Directory containing input files.
    - fn (str): File name of the input image.
    - output_dir (str): Directory to save the output.
    - unet_type (str): Type of U-Net model.
    - net (torch.nn.Module): Trained U-Net model.
    - scale_factor (float): Scale factor for image preprocessing.
    - out_threshold (float): Probability threshold for mask prediction.
    - device (torch.device): Computational device (CPU or GPU).
    - if_true_mask (bool): Whether to compare with true masks.
    - if_vis (bool): Whether to visualize the result.
    
    Returns:
    - np.ndarray: Predicted mask.
    """
    if fn.endswith('.npy'):
        fnp = os.path.join(in_files, fn)
        logging.info(f'\nPredicting image {fn} ...')
        img = np.load(fnp)

        radar_r = img[:, :, 0]
        radar_v = img[:, :, 1]
        radar_w = img[:, :, 2]
        radar_depomask = img[:, :, 3]

        # Predict mask based on image size
        if img.shape == (256, 256, 4):
            mask = predict_img(unet_type, net, img, scale_factor=scale_factor, out_threshold=out_threshold, device=device)
        else:
            mask = slide_window_predict(unet_type, net, img, scale_factor=scale_factor, out_threshold=out_threshold, device=device)

        # Post-process the mask
        echo_mask = np.full(mask.shape, 0)
        echo_mask[radar_r >= -50] = 1
        echo_mask[radar_v >= -15] = 1
        echo_mask[radar_w >= 0] = 1
        echo_mask[mask == 1] += 1
        echo_mask[:, 150:][echo_mask[:, 150:] == 1] = 2  # Post-process for higher altitudes

        # Save the prediction mask as a .npy file
        np.save(os.path.join(output_dir, fn), echo_mask)

        return echo_mask, range(len(img)), img


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Load the model based on U-Net type
    if args.unet_type == 'trans':
        net = TransUNet(img_dim=256, in_channels=4, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=1)
        args.model = r'model\TransUnet\CP_epoch60_miou_0.8984.pth'
    elif args.unet_type == 'se':
        net = UNetWithSE(n_channels=4, n_classes=1)
        args.model = r'model\UnetSE_LDR_randomDROP\CP_epoch60_miou_0.956.pth'
    else:
        net = UNet(n_channels=4, n_classes=1)
        args.model = r'model\Unet\CP_epoch60_miou_0.9766.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    # Process all files in the input directory
    files = os.listdir(args.input[0])
    for fn in files:
        mask, time, img = predict_met_echo(args.input[0], fn, args.output[0], args.unet_type, net, args.scale, args.mask_threshold, device)
        
        # Visualization and ground truth comparison
        r = img[:, :, 0]
        Height = np.array(range(len(r[0]))) * 0.03
        Time = np.array(range(len(r)))
        Visualize_type(mask, Height, Time, Title='Deep Learning', Height_range=(0, 15), Save_path=None, unit='', vmin=0, vmax=2, cbar_ticks=[0, 1, 2], cbar_labels=['None', 'CA E', 'Met E'])

        if os.path.exists(os.path.join(args.mask[0], fn)):
            met = np.load(os.path.join(args.mask[0], fn))
            mask_gt = (r >= -50).astype(int)
            mask_gt[met == 1] += 1
            Visualize_type(mask_gt, Height, Time, Title='Ground Truth', Height_range=(0, 15), Save_path=None, unit='', vmin=0, vmax=2, cbar_ticks=[0, 1, 2], cbar_labels=['None', 'CA E', 'Met E'])
