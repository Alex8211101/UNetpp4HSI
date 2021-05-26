import glob
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
# import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image

from dataset.ZHDataset import ZHDataset
from models.UNet_Nested import UNet_Nested
from utils import train_net

# from torch.cuda.amp import autocast


Image.MAX_IMAGE_PIXELS = 1000000000000000

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda")


import glob

import numpy as np
from torchsat.transforms import transforms_seg

train_transform = transforms_seg.Compose([
    transforms_seg.RandomVerticalFlip(p=0.5),
    transforms_seg.RandomHorizontalFlip(p=0.5),
    # transforms_seg.RandomShift(max_percent=0.4),
    # transforms_seg.RandomRotationY(),
])

train_path = './train_data'
imgs_dirs = sorted(glob.glob(os.path.join(train_path,"img*.tif")),key=os.path.getmtime)
# val_ratio = 0.2
random_state = 42
imgs_dirs = np.array(imgs_dirs)

# mass_dataset = ZHDataset(train_path = imgs_dirs, transform=train_transform)
# sample_nums = len(mass_dataset)
# sample_nums_train = sample_nums*(1-val_ratio)
# train_data, valid_data = torch.utils.data.random_split(mass_dataset, [int(sample_nums_train), sample_nums-int(sample_nums_train)])



model = UNet_Nested().cuda()

model_name = "UNet_Nested"
save_ckpt_dir = os.path.join('/checkpoints/', model_name, 'ckpt')
save_log_dir = os.path.join('/checkpoints/', model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)


param = {}

param['epochs'] = 50         
param['batch_size'] = 16     
param['lr'] = 1e-3            
param['gamma'] = 0.2          
param['step_size'] = 5        
param['momentum'] = 0.9       
param['weight_decay'] = 5e-4    
param['disp_inter'] = 1       
param['save_inter'] = 4       
param['iter_inter'] = 50     
param['min_inter'] = 10

param['model_name'] = model_name          
param['save_log_dir'] = save_log_dir      
param['save_ckpt_dir'] = save_ckpt_dir    
param['k_folds'] = 5

param['load_ckpt_dir'] = None


# if __name__ == '__main__':
best_models, models = train_net(param, model, imgs_dirs,train_transform, plot=True)
