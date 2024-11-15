import torch
import torch.nn.functional as F
from tqdm import tqdm
from loss.diceLoss import dice_coeff
from loss.bceLoss import BCE_loss
import math
import numpy as np
def calculate_iou(pred, target, n_classes):
    ious = []
    
    
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        #intersection = (pred_inds[target_inds]).long().sum().item()
        #交集
        intersection =np.sum (pred_inds & target_inds)
        #并集
        #union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        union =np.sum (pred_inds | target_inds)
        if union == 0:
            ious.append(float('nan'))  # 如果没有目标类，忽略该类
            '''elif cls==1:# 仅计算背景和目标类的IoU
            ious.append(intersection / union)'''
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

'''def eval_net(net, loader, device, n_val, n_classes):
    net.eval()
    tot = 0
    iou_scores = []

    with torch.no_grad():
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            masks_pred = net(imgs)
            if n_classes > 1:
                tot += F.cross_entropy(masks_pred, true_masks).item()
                pred = torch.argmax(masks_pred, dim=1).cpu().numpy()
                true = true_masks.cpu().numpy()
            else:
                pred = (torch.sigmoid(masks_pred) > 0.5).cpu().numpy().astype(int)
                true = true_masks.cpu().numpy().astype(int)
                tot += F.binary_cross_entropy_with_logits(masks_pred, true_masks).item()

            # 计算 IOU
            for i in range(pred.shape[0]):
                iou = calculate_iou( pred[i].flatten(),true[i].flatten(), 1)
                iou_scores.append(iou)

    avg_iou_score = sum(iou_scores) / len(iou_scores)
    print(f'Average IOU Score: {avg_iou_score}')
    return avg_iou_score'''

def eval_net(net, loader, device, n_val, n_classes):
    net.eval()
    tot_loss = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            masks_pred = net(imgs)
            if n_classes > 1:
                tot_loss += F.cross_entropy(masks_pred, true_masks).item()
                pred = torch.argmax(masks_pred, dim=1).cpu().numpy()
                true = true_masks.cpu().numpy()
            else:
                pred = (torch.sigmoid(masks_pred) > 0.5).cpu().numpy().astype(int)
                true = true_masks.cpu().numpy().astype(int)
                tot_loss += F.binary_cross_entropy_with_logits(masks_pred, true_masks).item()

            # 计算 Dice 系数和 IoU
            for i in range(pred.shape[0]):
                dice = dice_coefficient(torch.tensor(pred[i]), torch.tensor(true[i]))
                dice_scores.append(dice)

                iou = iou_coefficient(torch.tensor(pred[i]), torch.tensor(true[i]))
                iou_scores.append(iou)

    avg_dice_score = sum(dice_scores) / len(dice_scores)
    avg_iou_score = sum(iou_scores) / len(iou_scores)
    
    print(f'Average Dice Score: {avg_dice_score}')
    print(f'Average IoU Score: {avg_iou_score}')
    
    return  avg_iou_score

# Dice 系数计算函数
def dice_coefficient(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

# IoU 系数计算函数
def iou_coefficient(pred, target, smooth=1e-6):
    intersection = (pred * target).sum().float()
    union = (pred + target).sum().float() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()