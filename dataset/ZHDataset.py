import glob
import os

import numpy as np
from osgeo import gdal
from skimage import io
from torch.utils.data import Dataset


def read_img(filename):
    dataset=gdal.Open(filename) 

    im_width = dataset.RasterXSize  
    im_height = dataset.RasterYSize  
    im_bands = dataset.RasterCount   

    im_geotrans = dataset.GetGeoTransform()  
    im_proj = dataset.GetProjection() 
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset 

    return im_data
class ZHDataset(Dataset):

    def __init__(self, imgs_dirs, train=True,transform=None):



        tmp_list = imgs_dirs
        self.train = train
        self.mask_list = tmp_list
        # self.gt_path = gt_list
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):


        image = read_img(self.mask_list[idx])
        mask = read_img(self.mask_list[idx].replace('img','gt'))

        if self.transform:
            image = image.transpose(1, 2, 0)
            result_img, mask = self.transform(image, mask)
            image=result_img.transpose(2, 0, 1)

        sample = {"image": image, "label": mask}
        return sample



