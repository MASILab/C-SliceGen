
import os
import random
import time
import numpy as np

from scipy.ndimage import rotate,shift,zoom
from scipy import ndimage
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure,binary_dilation
from scipy.ndimage import gaussian_filter
from scipy.ndimage import measurements
from scipy.ndimage.measurements import label as lb
import torch 



class TransformCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, RandomRotates,Randomshift,RandomFlip,Rescale,ChangeIntensity):
        # self.transforms = transforms
        self.RandomRotates = RandomRotates
        self.Randomshift = Randomshift
        self.RandomFlip = RandomFlip
        self.Rescale = Rescale
        self.ChangeIntensity = ChangeIntensity
        

    def __call__(self, x_t,x_c,seg):

        if self.RandomRotates is not 0:
            x_t,x_c,seg = self._RandomRotates(x_t,x_c,seg)
        if self.Randomshift is not 0:
            x_t,x_c,seg = self._Randomshift(x_t,x_c,seg)
        if self.RandomFlip is not 0:
            x_t,x_c,seg = self._RandomFlip(x_t,x_c,seg)
        if self.Rescale != 512:
            scale = int(self.Rescale)/512
            x_t,x_c,seg = self._Rescale(x_t,x_c,seg,scale)
        # if self.ChangeIntensity is not 0:
        #     x_t,x_c,seg = self._ChangeIntensity(x_t,x_c,seg)
        return x_t,x_c,seg

    def _RandomFlip(self,x_t,x_c,seg,p=0.5):
        if random.random() < p:
            if len(x_t.shape) == 3:
                x_t = np.rot90(x_t,k=1,axes=(1,2))
            else:
                x_t = np.rot90(x_t)
            x_c = np.rot90(x_c)
            if seg is not None:
                seg = np.rot90(seg)
            # print(type(x_t), type(x_c), type(seg))
        return x_t.copy(), x_c.copy(), seg#.copy()
            # else:
            #     return x_t.copy(), x_c.copy(), seg

    def _RandomRotates(self,x_t,x_c,seg,degree=45):
        rotate_degree = random.random() * 2 * degree - degree
        if len(x_t.shape) == 3:
            x_t = rotate(x_t, rotate_degree, order=1, axes=(1, 2), reshape=False, cval=0.0, prefilter=False)
        else:
            x_t = rotate(x_t, rotate_degree, order=1, reshape=False, cval=0.0, prefilter=False)
        x_c = rotate(x_c, rotate_degree, order=1, reshape=False, cval=0.0, prefilter=False)
        if seg is not None:
            seg = rotate(seg, rotate_degree, order=0, reshape=False, cval=0.0, prefilter=False)
        return x_t, x_c, seg

    def _Randomshift(self,x_t,x_c,seg,s=10):
        shiftV = random.random() * s
        if len(x_t.shape) == 3:
            x_t = shift(x_t, (0, shiftV, shiftV), order=1, cval=0.0, prefilter=False)
        else:
            x_t = shift(x_t, shiftV, order=1, cval=0.0, prefilter=False)
        x_c = shift(x_c, shiftV, order=1, cval=0.0, prefilter=False)
        if seg is not None:
            seg = shift(seg, shiftV, order=0, cval=0.0, prefilter=False)
        return x_t, x_c, seg

    def _Rescale(self,x_t,x_c,seg, scale):
        if len(x_t.shape) == 3:   
            x_t = zoom(x_t, (1, scale,scale), order=1)
        else:
            x_t = zoom(x_t, scale, order=1)
        x_c = zoom(x_c, scale, order=1)
        if seg is not None:
            seg = zoom(seg, scale, order=0)

        return x_t,x_c,seg

    def _ChangeIntensity(self, x_t, x_c, seg, p=0.5):
        if random.random() < p:
            intensity = np.random.uniform(low=0.8,high=1.2) 
            x_t = x_t * intensity
            x_c = x_c * intensity

        return x_t, x_c, seg