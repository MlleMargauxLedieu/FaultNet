# Train
# TODO: Add logger


import os, sys
from collections import defaultdict
from copy import deepcopy
import argparse
import logging
import datetime
import time
import numpy as np
#import json
import csv
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.autograd import Variable

from loaders import dataloader 
from utils.loss import dice_random_weight, cross_entropy_3D, CE_plus_Dice
from models import load_models

def compute_iou(preds, trues):

    b,c,_,_,_ = preds.shape
    #print('---Preds---', np.amax(preds))
    #print('---Trues---', np.amax(trues))
    if c==2:
       preds = np.argmax(preds, axis=1)
    preds = preds.reshape(b,-1)
    trues = trues.reshape(b,-1)
    preds = preds>0.5
    trues = trues>0.5
    inter = trues & preds
    union = trues | preds
    iou = inter.sum(1)/(union.sum(1)+1e-8)
    return iou

def check_folder(data_path,file_ext):
    """
      helper function to check all training data paths are correct
    """

    # Check if all images and labels exist
    if not os.path.exists(data_path):
        error_message = 'Folder ' + data_path + ' not found'
        raise OSError(error_message)

    image_path = os.path.join(data_path, 'images')
    label_path = os.path.join(data_path, 'labels')

    if not os.path.exists(image_path):
        error_message = 'Folder ' + image_path + ' does not exist'
        raise OSError(error_message)

    if not os.path.exists(label_path):
        error_message = 'Folder ' + label_path + ' does not exist'
        raise OSError(error_message)

    # Build a list of all images and labels
    image_list = []
    label_list = []

    for image in os.listdir(image_path):
        if image.endswith(file_ext):
            image_list.append(image.split('.')[0])

    for label in os.listdir(label_path):
        if label.endswith(file_ext):
            label_list.append(label.split('.')[0])

    image_list.sort()
    label_list.sort()

    if not image_list == label_list:
        print('Lists are different')
        
        file_list = list(set(image_list) & set(label_list))
    else:
        file_list = image_list

    print("There were {} images in the training set.".format(len(file_list)))
    return file_list

def train_model(model, criterion, optimizer, lr_scheduler, nepochs):
   
    return model

def main(args):

    training_data_path = args.training_path
    output_path = args.model_name

    model_arch = args.model_arch
    model_path = args.model_path
    batch_size = args.batch_size
    nepoch = args.epoch_count
    initial_lr = args.learning_rate

    output_channels = 2

    print(" ---model will be written at %s ---  "%(output_path))
    print("----- model arch is %s -----  "%(model_arch))
    print("----- output_channels %d "%(output_channels))

    file_list_train = check_folder(training_data_path, '.npy')
    training_data_loader, validation_data_loader = dataloader.LoadData(training_data_path, file_list_train, split=0.001, \
                                                                             batch_size=batch_size, transforms=None)

    parallel_flag=True
    gpu_flag=True
    model = load_models.getModel(model_arch=model_arch,output_channels=output_channels,parallel_flag=parallel_flag)
    curr_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['model_state']  # this is wrapped in DataParallel so need to undo it
    model_dict = OrderedDict()
    # for k, v in curr_dict.items():
    #     if 'module' not in k:
    #        name = 'module.' +k
    #    else:
    #        name = k[7:]
    #    model_dict[name] = v
    model.load_state_dict(curr_dict)
    print("-----training loader  length is ---------  ",len(training_data_loader))
    print("----- val loader length is -------------  ",len(validation_data_loader))


    step_decay = 30
    decay_factor = 0.5
    optimizer = torch.optim.Adam(model.parameters(),lr=initial_lr, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_decay, decay_factor) ## decay every 30 epochs by half

    LossFunction = cross_entropy_3D  # default loss, change if needed

    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_iou = -100.0
    for epoch in range(nepoch):
        print('Epoch {}/{}'.format(epoch, nepoch - 1))
        print('-' * 10)

        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = []
            for i, (images, labels) in enumerate(training_data_loader):
                images = images.float().cuda()
                labels = labels.long().cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred_dict = model(images)
                    loss = LossFunction(pred_dict, labels,1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss +=loss.item()
                running_corrects.append(compute_iou(pred_dict['logits'].detach().cpu().numpy(), \
                                                labels.detach().cpu().numpy())) 
            if phase == 'train':
                lr_scheduler.step()
            
            epoch_loss = running_loss/len(training_data_loader)
            #print('---- Running corrects ---', running_corrects)
            epoch_iou = np.concatenate(running_corrects).mean()
            print('{} Loss: {:.4f} IOU: {:.4f}'.format(
                phase, epoch_loss, epoch_iou))
            # deep copy the model
            
            if phase == 'validate' and epoch_iou > best_iou:
                #print('Improved the score from {:.4f} to {:.4f}, saving model..'.format(epoch_iou, best_iou))
                best_iou = epoch_iou
                best_model = copy.deepcopy(model.state_dict())
                save_state = {'epoch': epoch+1,
                              'model_state': model.state_dict(),
                              'optimizer_state': optimizer.state_dict()}
                modelfile = model_arch + '_raw_10_20Hz_synth.pkl'
                save_filename =  os.path.join(output_path,modelfile)
                print('Saving model at ', save_filename)
                torch.save(save_state,  save_filename)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_iou))

        # load best model weights
    model.load_state_dict(best_model)

if __name__=="__main__":
    help_string = "PyTorch Fault picking Model Training"

    parser = argparse.ArgumentParser(description=help_string)

    parser.add_argument('-t', '--training-path', type=str, metavar='DIR', help='Path where training data exists', required=True)
    parser.add_argument('-vl', '--validation-path', type=str, metavar='DIR', help='Path where validation data exists', required=False)  #TODO
    parser.add_argument('-n', '--model-name', type=str, metavar='DIR', help='Path of model to be stored', required=True)
    parser.add_argument('-m', '--model-path', type=str, metavar='DIR', help='Path where trained model is stored', required=True)

    parser.add_argument('-bs', '--batch-size', type=int, metavar='N', help='Batch size', default=8, required=True)
    parser.add_argument('--epoch-count', type=int, metavar='N', help='Number of epochs (default: 100)', default=100, required=True)
    parser.add_argument('-lr', '--learning-rate', type=float, metavar='LR', help='Initial learning rate (default: 0.0005)', default=0.0005, required=True)
    parser.add_argument('--validation-split', type=float, metavar='VAL', help='Training-validation split (default: 0.15)', default=0.15, required=False)

    parser.add_argument('-arch', '--model-arch', type=str, metavar='ARCH', help='Architecture of the model ', default='vnet', required=True)

    args = parser.parse_args()

    main(args)
    sys.exit(1)
