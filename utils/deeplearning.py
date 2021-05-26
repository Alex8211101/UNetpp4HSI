import copy
import logging
import os
import random
import time
from glob import glob

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations import functional as F
from dataset.ZHDataset import ZHDataset
from PIL import Image
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import (DiceLoss, FocalLoss,
                                                SoftBCEWithLogitsLoss,
                                                SoftCrossEntropyLoss)
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast  # need pytorch>1.6
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional
from tqdm import tqdm

from utils.utils import AverageMeter, inial_logger, second2time

from .metric import IOUMetric

Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_net(param, model, imgs_dirs,train_transform,plot=False,device='cuda'):

    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    lr              = param['lr']
    gamma           = param['gamma']
    step_size       = param['step_size']
    momentum        = param['momentum']
    weight_decay    = param['weight_decay']
    k_folds         = param['k_folds']

    disp_inter      = param['disp_inter']
    save_inter      = param['save_inter']
    min_inter       = param['min_inter']
    iter_inter      = param['iter_inter']

    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']

    #
    scaler = GradScaler() 
      # For fold results
    results = {}
    models = {}
    best_modes = {}
  
    # Set fixed random number seed
    # torch.manual_seed(42)
    kfold = KFold(n_splits=k_folds, shuffle=True,random_state=42)
    # sample_nums = len(mass_dataset)
    # sample_nums_train = sample_nums*(1-val_ratio)
    # train_data, valid_data = torch.utils.data.random_split(mass_dataset, [int(sample_nums_train), sample_nums-int(sample_nums_train)])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(imgs_dirs)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataset = ZHDataset(imgs_dirs = imgs_dirs[train_ids], train=True,transform=train_transform)
        valid_dataset = ZHDataset(imgs_dirs = imgs_dirs[test_ids], train=False,transform=None)

        
        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data_size = train_dataset.__len__()
        valid_data_size = valid_dataset.__len__()
        c, y, x = train_loader.__getitem__(0)['image'].shape
        # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        # valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
        optimizer = optim.AdamW(model.parameters(), lr=lr ,weight_decay=weight_decay)
        #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
        #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
        #criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        DiceLoss_fn=DiceLoss(mode='multiclass')

        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                                first_weight=0.5, second_weight=0.5).cuda()
        logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) +'_'+model_name+ '.log'))


        train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
        train_loader_size = train_loader.__len__()
        valid_loader_size = valid_loader.__len__()
        best_iou = 0
        best_epoch=0
        best_mode = copy.deepcopy(model)
        epoch_start = 0
        if load_ckpt_dir is not None:
            ckpt = torch.load(load_ckpt_dir)
            epoch_start = ckpt['epoch']
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
        #

        for epoch in range(epoch_start, epochs):
            epoch_start = time.time()

            model.train()
            train_epoch_loss = AverageMeter()
            train_iter_loss = AverageMeter()
            for batch_idx, batch_samples in enumerate(train_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long))
                # with autocast(): #need pytorch>1.6
                pred = model(data)
                # print(pred.shape)
                # print(target.shape)
                loss = criterion(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(epoch + batch_idx / train_loader_size) 
                image_loss = loss.item()
                train_epoch_loss.update(image_loss)
                train_iter_loss.update(image_loss)
                if batch_idx % iter_inter == 0:
                    spend_time = time.time() - epoch_start
                    logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                        epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                        optimizer.param_groups[-1]['lr'],
                        train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                    train_iter_loss.reset()


            model.eval()
            valid_epoch_loss = AverageMeter()
            valid_iter_loss = AverageMeter()
            iou=IOUMetric(10)
            with torch.no_grad():
                for batch_idx, batch_samples in enumerate(valid_loader):
                    data, target = batch_samples['image'], batch_samples['label']
                    data, target = Variable(data.to(device,dtype=torch.float)), Variable(target.to(device,dtype=torch.long))
                    pred = model(data)
                    # pred = model(data[:,:,:,12],data[:,:,:,0:12])
                    loss = criterion(pred, target)
                    pred=pred.cpu().data.numpy()
                    pred= np.argmax(pred,axis=1)
                    iou.add_batch(pred,target.cpu().data.numpy())
                    #
                    image_loss = loss.item()
                    valid_epoch_loss.update(image_loss)
                    valid_iter_loss.update(image_loss)
                    # if batch_idx % iter_inter == 0:
                    #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                    #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
                val_loss=valid_iter_loss.avg
                acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
                logger.info('[val] epoch:{} miou:{:.4f}'.format(epoch,mean_iu))
                    


            train_loss_total_epochs.append(train_epoch_loss.avg)
            valid_loss_total_epochs.append(valid_epoch_loss.avg)
            epoch_lr.append(optimizer.param_groups[0]['lr'])

            if epoch % save_inter == 0 and epoch > min_inter:
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}-fold{}.pth'.format(epoch,fold))
                torch.save(state, filename)  

            if mean_iu > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                filename = os.path.join(save_ckpt_dir, 'checkpoint-best-{}.pth'.format(fold))
                torch.save(state, filename)
                best_iou = mean_iu
                best_mode = copy.deepcopy(model)
                logger.info('[save] Best Model saved at epoch:{} fold:{} ============================='.format(epoch,fold))
            scheduler.step()

        if plot:
            x = [i for i in range(epochs)]
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
            ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('CrossEntropy', fontsize=15)
            ax.set_title('train curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(x, epoch_lr,  label='Learning Rate')
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Learning Rate', fontsize=15)
            ax.set_title('lr curve', fontsize=15)
            ax.grid(True)
            plt.legend(loc='upper right', fontsize=15)
            plt.savefig("learning rate.png", dpi = 300)
            plt.show()

        results[fold] =  100*best_iou       
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(results.items())} %')
        models[fold] = model
        best_modes[fold] = best_mode

    return best_modes, models
