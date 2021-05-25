import glob
import os

import numpy as np
from osgeo import gdal
from skimage import io
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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



def write_img(im_data,filename):

    #gdal.GDT_Byte, 
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64


    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32


    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 


    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)



    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

train_path = "./comp/train/"
# for parent,tile,_ in os.walk(train_path):
#     tiles = [os.path.join(train_path,p) for p in tile]
#     break

tmp_list = []
gt_list = []

for i in range(1,100):
    idx = str(i).zfill(4)+'.tif'
    if(i%10) == 0:
        continue
    tmp_list.append(os.path.join(train_path,'images',idx))
    gt_list.append(os.path.join(train_path,'labels',idx))


save_ckpt_dir = "./train_data"
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)

idx = 0
for im_path ,gt_path in tqdm(zip(tmp_list,gt_list)):

    # maskpath = self.mask_list[idx]
    image = read_img(im_path)
    image = image/1000.0
    image = np.array(image,'float')



    label = read_img(gt_path)

    label[label==255] = 0
    # label = resize(label,(500,500),order=0,mode='edge',preserve_range=True)
    label = np.array(label,'uint8')
    # 500*500->256*256 stride = 128



    for i in range(2):
        for j in range(2):
            x_patch = image[:,i*128:(i+2)*128,j*128:(j+2)*128]
            y_patch = label[i*128:(i+2)*128,j*128:(j+2)*128]

            if(j==2 and i!=2):
                x_patch = image[:,i*128:(i+2)*128,500-256:500]
                y_patch = label[i*128:(i+2)*128,500-256:500]

            if(j!=2 and i==2):
                x_patch = image[:,500-256:500,j*128:(j+2)*128]
                y_patch = label[500-256:500,j*128:(j+2)*128]
            
            if(j==2 and i==2):
                x_patch = image[:,500-256:500,500-256:500]
                y_patch = label[500-256:500,500-256:500]

            
            write_img(x_patch,os.path.join(save_ckpt_dir,"img_"+str(idx).zfill(4)+".tif"))
            write_img(y_patch,os.path.join(save_ckpt_dir,"gt_"+str(idx).zfill(4)+".tif"))
            idx+=1

print("end")
