import os
import sys
import logging
import torch
import numpy as np

from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, unet_type, imgs_dir, masks_dir, scale=1):
        self.unet_type = unet_type
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        files_sv=os.listdir(imgs_dir)
        print(len(files_sv))
        #去掉文件名后缀
        files_sv=[f.split('.')[0] for f in files_sv]
        self.ids = files_sv
        print(len(files_sv))
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess(cls, unet_type, pil_img, scale):
        #print(pil_img.shape)
        w, h = pil_img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        if unet_type != 'v3':
            #pil_img = pil_img.resize((newW, newH))
            #narray resize
            if len(pil_img.shape) == 2:
                pil_img = np.resize(pil_img, (newW, newH))
            else:
                pil_img = np.resize(pil_img, (newW, newH, pil_img.shape[2]))
        else:
            new_size = int(scale * 640)
            #pil_img = pil_img.resize((new_size, new_size))
            if len(pil_img.shape) == 2:
                pil_img = np.resize(pil_img, (new_size, new_size))
            else:
                pil_img = np.resize(pil_img, (new_size, new_size, pil_img.shape[2]))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        #print(img_trans.shape)
        '''if img_trans.max() > 1:
            img_trans = img_trans / 40.0'''
        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file_path=os.path.join(self.masks_dir, idx + '.npy')
        mask_file = mask_file_path
        img_file_path=os.path.join(self.imgs_dir, idx + '.npy')
        img_file = img_file_path
        #print(mask_file,img_file)
        #assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file) #Image.open(mask_file[0])
        #mask = mask [:,:,1:]
        #mask one-hot
        if len(mask.shape)==3:
            if mask.shape[2] == 3:
                mask = mask[:,:,1]
        '''else:
            mask=mask.reshape(mask.shape[0],mask.shape[1],1)'''
            

        #增加一个维度 w,h -> w,h,1
        #mask = np.expand_dims(mask, axis=2)
        #mask = np.moveaxis(mask, -1, 0)


        img = np.load(img_file)
        #随机将第四个通道置为nan  概率为0.25
        if np.random.rand() < 0.25:
            img[:,:,3] = np.nan
        img = np.array(img, dtype=np.float32)
        #nan处理为-99
        img[np.isnan(img)] = -99
        #inf处理为-99
        img[np.isinf(img)] = -99
        #print(mask.shape,img.shape)
        assert img.shape[:2] == mask.shape[:2], f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'
        #转为img
        #img = Image.fromarray(img)
        #mask = Image.fromarray(mask)
        #print(img.shape,mask.shape)
        
        # 确保图像和掩码的大小一致
        target_size = (256, 256)  # 你可以根据需要调整这个大小
        #print(img.shape,mask.shape)

        img = self.preprocess(self.unet_type, img, self.scale)
        mask = self.preprocess(self.unet_type, mask, self.scale)
        
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
