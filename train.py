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
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet3D, UNet, UNet2Plus, UNet3Plus, TransUNet, UNetWithSE
from utils.dataset import BasicDataset
from utils.eval import eval_net
from torchsummary import summary
from torch.utils.data._utils.collate import default_collate
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np 
import matplotlib.pyplot as plt

# Set environment variables
torch.cuda.empty_cache()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Directories for datasets
dir_img_test = r'F:\Workspace\Projects\气象局技能大赛\基于机器学习的晴空回波识别\Data_test_400/imgs'
dir_mask_test = r'F:\Workspace\Projects\气象局技能大赛\基于机器学习的晴空回波识别\Data_test_400/masks'
dir_img = r'F:\Workspace\Projects\气象局技能大赛\基于机器学习的晴空回波识别\Data_adjusted_psahu_400ag\imgs'
dir_mask = r'F:\Workspace\Projects\气象局技能大赛\基于机器学习的晴空回波识别\Data_adjusted_psahu_400ag\masks'
dir_checkpoint = r'400_r2_seunet_drop2/'

# Create checkpoint directory if it doesn't exist
os.makedirs(dir_checkpoint, exist_ok=True)

# Custom collate function for DataLoader to handle batch processing
def custom_collate_fn(batch):
    new_batch = []
    for item in batch:
        new_item = {}
        for key, value in item.items():
            new_item[key] = torch.tensor(value, dtype=torch.float32) if isinstance(value, np.ndarray) else value.float()
        new_batch.append(new_item)
    return default_collate(new_batch)

# Training function
def train_net(unet_type, net, device, epochs=5, batch_size=1, lr=0.1, save_cp=True, img_scale=0.5):
    scaler = GradScaler()
    dataset = BasicDataset(unet_type, dir_img, dir_mask, img_scale)
    dataset_test = BasicDataset(unet_type, dir_img_test, dir_mask_test, img_scale)

    n_val = len(dataset_test) 
    n_train = len(dataset)

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f"Starting training with the following parameters:\n"
                 f"UNet type: {unet_type}\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {lr}\n"
                 f"Dataset size: {n_val + n_train}\nTraining size: {n_train}\nValidation size: {n_val}\n"
                 f"Checkpoints: {save_cp}\nDevice: {device.type}\nImages scaling: {img_scale}")

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)

    # Learning rate scheduler (Cosine Annealing)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # Define loss functions
    criterion = nn.BCEWithLogitsLoss() if net.n_classes == 1 else nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0, 4.0]))

    # Training loop
    lrs = []
    best_loss = float('inf')
    best_val_score = 0

    for epoch in range(epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch={epoch + 1} lr={cur_lr}')
        net.train()
        epoch_loss = 0

        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float16 if net.n_classes == 1 else torch.long)

                masks_pred = net(imgs)
                optimizer.zero_grad()

                # Automatic mixed precision
                with autocast():
                    loss = criterion(masks_pred, true_masks)

                if torch.isnan(loss):
                    print("Loss is NaN. Skipping this batch.")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix({'loss (batch)': loss.item()})

                pbar.update(imgs.shape[0])
                global_step += 1

        # Validation
        if epoch % 2 == 0:
            val_score = eval_net(net, val_loader, device, n_val, net.n_classes)
            logging.info(f'Validation score: {val_score}')
            writer.add_scalar('Dice/test' if net.n_classes == 1 else 'Loss/test', val_score, global_step)

            if val_score > best_val_score or epoch == epochs - 1:
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}_miou_{val_score}.pth'))
                best_val_score = val_score
                logging.info(f'Checkpoint {epoch + 1} saved!')

        scheduler.step()
        lrs.append(cur_lr)
    # Plot learning rate schedule
    plt.plot(lrs, '.-', label='LambdaLR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=300)
    writer.close()

# Freezing layers of the model
def freeze_layers(model, num_layers_to_freeze, StartingLayer=0):
    layers = list(model.children())
    for layer in layers[StartingLayer:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('-u', '--unet_type', dest='unet_type', type=str, default='se', help='UNet type (v1/v2/v3)')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('-f', '--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', type=float, default=1, help='Downscaling factor of the images')
    return parser.parse_args()

# Main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Initialize the model based on the selected UNet type
    if args.unet_type == 'v2':
        net = UNet2Plus(n_channels=4, n_classes=1)
    elif args.unet_type == 'v3':
        net = UNet3Plus(num_classes=1)
    elif args.unet_type == '3d':
        net = UNet3D(in_channels=4, out_channels=1)
    elif args.unet_type == 'trans':
        net = TransUNet(img_dim=256, in_channels=4, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=1)
    elif args.unet_type == 'se':
        net = UNetWithSE(n_channels=4, n_classes=1)
    else:
        net = UNet(n_channels=4, n_classes=1)

    # Freeze the initial layers
    freeze_layers(net, num_layers_to_freeze=5)

    # Load the model if specified
    if args.load:
        logging.info(f'Loading model {args.load}')
        net.load_state_dict(torch.load(args.load))

    net.to(device=device)

    # Start training
    train_net(args.unet_type, net, device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,save_cp=True, img_scale=args.scale)
