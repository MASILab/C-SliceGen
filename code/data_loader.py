from cmath import phase
import numpy as np
import torch
from torch.utils.data import Dataset
import os, pickle, csv
import time
import collections
import random
#from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
#from layers import nms,iou
import pandas
from imgaug import augmenters as iaa
import imgaug as ia
import json
import time
from PIL import Image
import pandas as pd
import nibabel as nb
from functional import *

class ImgAugTransform_train:
    def __init__(self, image_size):
        self.aug = iaa.Sequential([
        iaa.Scale((image_size, image_size)),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-40, 40), mode='symmetric'),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])
    
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
    
class ImgAugTransform_val:
    def __init__(self, image_size):
        self.aug = iaa.Sequential([
        iaa.Scale((image_size, image_size)),
    ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
    
data_aug_train =  {
            'RandomRotates': 45,
            'Randomshift': 10,
            'RandomFlip': 0.5,
            'Rescale':256,
            'ChangeIntensity':0.5
            }

data_aug_test =  {
            'RandomRotates': 0,
            'Randomshift': 0,
            'RandomFlip': 0,
            'Rescale':256,
            'ChangeIntensity':0
            }
            

def rescale(image, scale):
    new_img = zoom(image, scale, order=1)
    return new_img



class abdomen_slice_pair(Dataset): 
    def __init__(self, config, phase = 'train'):

        self.df_pair = pd.read_csv(config['pair_csv'])
        self.imglist = self.df_pair.loc[(self.df_pair['phase']==phase), 'first image'].values
        
        self.img_size = config['img_size']
        self.datadir = {'btcv':config['btcvdatadir'], 'imagevub':config['imagevubdatadir'], 'imagevub2':config['imagevub2datadir']}

        self.phase = phase

        self.transform_train = TransformCompose(**data_aug_train)
        self.transform_val = TransformCompose(**data_aug_test)
       
        self.data_aug = config['data_aug']
        self.config = config
    

        
    def __getitem__(self, idx, split=None):
        
        t = time.time()
        
        np.random.seed(int(str(t%1)[2:7]))  
        cond_filename = self.imglist[idx]  
        target_filename = self.df_pair.loc[(self.df_pair['first image']==cond_filename), 'second image'].values[0] 
        dataset = self.df_pair.loc[(self.df_pair['first image']==cond_filename), 'dataset'].values[0] 

    
        if self.phase == 'val':
            target_img = np.array(Image.open(os.path.join(self.datadir[dataset]['train'], target_filename)).convert('L')).astype(float) # load img
            cond_img = np.array(Image.open(os.path.join(self.datadir[dataset]['train'], cond_filename)).convert('L')).astype(float) # load img
        else:
            target_img = np.array(Image.open(os.path.join(self.datadir[dataset][self.phase], target_filename)).convert('L')).astype(float) # load img
            cond_img = np.array(Image.open(os.path.join(self.datadir[dataset][self.phase], cond_filename)).convert('L')).astype(float) # load img
        
        target_img /= 255. # convert to [0,1]
        target_img = (target_img - target_img.min()) * 1.0 / (target_img.max() - target_img.min()) # normalize

    
        # cond_img = nb.load(os.path.join(self.datadir['conditional'], self.phase, cond_filename.replace('.png', '.nii.gz'))).get_fdata()
        
        cond_img /= 255. # convert to [0,1]
        cond_img = (cond_img - cond_img.min()) * 1.0 / (cond_img.max() - cond_img.min()) # normalize


        if self.data_aug:
            if self.phase == 'train':
                target_img, cond_img, _ = self.transform_train(target_img, cond_img, cond_img)
            else:
                target_img, cond_img, _ = self.transform_val(target_img, cond_img, cond_img)
        else:
            target_img = rescale(target_img, 0.5)
            cond_img = rescale(cond_img, 0.5)
        
        target_img = target_img[np.newaxis,...]
        cond_img = cond_img[np.newaxis,...]
        # target_label = target_label[np.newaxis,...]
        # print(target_filename, cond_filename)

        return torch.from_numpy(cond_img).float(), torch.from_numpy(target_img).float(), cond_filename
            
        
    def __len__(self):
        return len(self.imglist)



class abdomen_blsa(Dataset): 
    def __init__(self, config, phase = 'train'):
     
        self.datadir = config['blsadatadir']
       
        self.imglist = os.listdir(self.datadir)
        
        self.img_size = config['img_size']
       
        self.phase = 'test'

        self.transform_val = TransformCompose(**data_aug_test)
       
        self.data_aug = config['data_aug']
        self.config = config
   
        
    def __getitem__(self, idx, split=None):
        
        cond_filename = self.imglist[idx]  
        print(self.datadir)
        
        cond_img = np.array(Image.open(os.path.join(self.datadir, cond_filename)).convert('L')).astype(float) # load img
       
        cond_img /= 255. # convert to [0,1]
        cond_img = (cond_img - cond_img.min()) * 1.0 / (cond_img.max() - cond_img.min()) # normalize

        cond_img = rescale(cond_img, 0.5)
        
        cond_img = cond_img[np.newaxis,...]

        return torch.from_numpy(cond_img).float(), cond_filename
         
    def __len__(self):
        return len(self.imglist)
