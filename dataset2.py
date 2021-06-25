import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from util.color_constancy import max_rgb
from util.reduce_method import remove_function_1,remove_function_3

class MMDataset(Dataset):
    def __init__(self, args, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = args.use_meta
        self.DANN = args.DANN
        self.transform = transform
        if args.out_dim==2:
            self.mel=1
        elif args.out_dim==9:
            self.mel=1
        self.num=0
        self.meta_columns=['age_approx', 'n_images', 'image_size'] + [col for col in csv.columns if col.startswith('site_')]
        self.cc = args.cc
        cc_methods={
            'max_rgb':max_rgb
        }
        self.cc_model = cc_methods[args.cc_method]
        self.remove = args.remove
        rm_methods={
            'rm1':remove_function_1,
            'rm3':remove_function_3
        }
        self.rm_model = rm_methods[args.remove_method]
    def __len__(self):
        return self.csv.shape[0]
    
    def get_num(self):
        
        num_mm=self.csv.loc[(self.csv['target'] == self.mel)].shape[0]
        num_normal=self.csv.loc[(self.csv['target'] != self.mel)].shape[0]
        return num_normal, num_mm, num_normal, num_mm
    
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath) # 默认读出的是BGR模式
        if self.cc:
            image = self.cc_model(image)
        if self.remove:
            image = self.rm_model(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # m*n*3
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32) # 512*512*3
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1) # 3*512*512

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_columns]).float())
        else:
            data = torch.tensor(image).float()
        if self.mode == 'test':
            return data
        elif self.DANN:
            return data, (torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(0).long())
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()
        
def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

# def get_df(args):
#     df_train = pd.read_csv(os.path.join(args.label_dir, f'train_ISIC2019.csv'), dtype={'image_name':str})
    
#     df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(args.label_dir, 'images/ISIC2019/', f'ISIC_2019_Training_Input_1024{x}.jpg'))
#     _idx=1
    
#     return df_train,None, _idx
def get_df(args):
    df_train = pd.read_csv(os.path.join(f'train_PH2.csv'), dtype={'image_name':str})
    
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join('PH2Dataset', 'images', f'{x}/{x}_Dermoscopic_Image/{x}.bmp'))
    _idx=1
    
    return df_train,None, _idx
