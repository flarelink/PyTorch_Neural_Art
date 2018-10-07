# =============================================================================
# art.py - Neural art in PyTorch
# Copyright (C) 2018  Humza Syed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import argparse
from datetime import datetime
import random
import io
import requests
from PIL import Image
plt.ion()


# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device, " the GPU device is selected")
else:
    device = torch.device('cpu')
    print(device, " the CPU device is selected")
    


def load_imgs(targ_img_loc, targ_on, style_img_loc, style_on):
    """
    Load target and style images
    :param targ_img_loc:   target img to have neural art applied to;
                           will either be a URL or location to the img
    :param targ_on:        determine if image is from online or local
                           to determine how to load img
    :param style_img_loc:  style img to have neural art used with;
                           will either be a URL or location to the img
    :param style_on:       determine if image is from online or local
                           to determine how to load img
    :return: loading images
    """

    # transform data
    # resize img down to 256 for crop
    # center crop to 224 for VGG model
    # to tensor for GPU processing
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    # create pillow image from target image
    if(targ_on):
      targ_img = requests.get(targ_img_loc)
      pil_targ = Image.open(io.BytesIO(targ_img.content))
    else:
      pil_targ = Image.open(targ_img_loc)

    # create pillow image from style image
    if(style_on):
      style_img = requests.get(style_img_loc)
      pil_style = Image.open(io.BytesIO(style_img.content))
    else:
      pil_style = Image.open(style_img_loc)

    # apply transforms to imgs
    targ_trans = trans(pil_targ)
    print(targ_trans.size())
    style_trans = trans(pil_style)
    print(style_trans.size())

    return targ_trans, style_trans



"""
##############################################################################
# CNN classes for network 
##############################################################################
"""
class CNN_1Lay_FullTrain(nn.Module):
    """
    CNN with 1 conv layer fully trained and 1 fully connected layer
    """
    # Convolutional neural network (two convolutional layers)
    def __init__(self, nf=64, filter_d=3, weights=None, in_chan=1, num_classes=10):
        super(CNN_1Lay_FullTrain, self).__init__()

        # CNN creation
        if (weights is None):
            print('weight type not specified')

        self.conv1 = nn.Conv2d(in_channels=in_chan,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv1.weight = nn.Parameter(weights)
        self.conv1.weight.requires_grad = True

        self.fc = nn.Linear(28 * 28 * nf, num_classes)  # 10 classes

        # print('The output size would be: {:d}'.format(outputSize(28, 3, 1, 1)))

    # CNN forward pass
    def forward(self, x):
        c1 = self.conv1(x)
        x = F.relu(c1)
        # print(x.shape)
        x = x.view(-1, 28 * 28 * 64)
        x = self.fc(x)
        return x

def create_parser():
  """
  return: parser inputs
  """
  parser = argparse.ArgumentParser(
        description='PyTorch Neural Art.')

  # function to allow parsing of true/false boolean
  def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
    
  # arguments for logging
  parser.add_argument('--log_name', type=str, default='testing',
                        help='Specify log name, if not specified then program will default to testing log name; default=testing')

  # arguments for target and style image locations
  parser.add_argument('--targ_img_loc', type=str, default='https://i.ytimg.com/vi/I7jgu-8scIA/maxresdefault.jpg', # cute cat img default :3
                        help='Determine location of targ img, whether online or not; default=https://i.ytimg.com/vi/I7jgu-8scIA/maxresdefault.jpg')
  parser.add_argument('--targ_on', type=str2bool, nargs='?', default=True,
                        help='Flag to determine if target image is from online link or not; default=True')

  parser.add_argument('--style_img_loc', type=str, default='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg', # cute cat img default :3
                        help='Determine location of style img, whether online or not; default=https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
  parser.add_argument('--style_on', type=str2bool, nargs='?', default=True,
                        help='Flag to determine if style image is from online link or not; default=True')
  
  # arguments for hyperparameters
  parser.add_argument('--n_epochs', type=int, default=50,
                        help='Defines num epochs for training; default=50')
  parser.add_argument('--lr', type=float, default=0.001,
                        help='Defines learning rate for training; default=0.001')
  parser.add_argument('--random_seed', type=int, default=7,
                        help='Defines random seed value; default=7')

  args = parser.parse_args()

  return args


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
if __name__ == '__main__':
    """
    Runs through two images iteratively to make neural artwork~
    """
    
    # parsing input arguments
    args = create_parser()

    # double check random seed
    if (args.random_seed == None):
         args.random_seed = random.randint(1, 1000)

    # set reproducible random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # make logs directory if it doesn't exist
    if(os.path.exists("./logs/") == False):
      os.mkdir('logs')
    

    # open log files and fill in hyperparameters
    today_date = str(datetime.today()).replace(':', '_').replace(' ', '_')
    log_file = open('./logs/log_run_{}_{}.txt'.format(today_date, args.log_name), 'w+')
    log_file.write('num_epochs     = {:d} \n'.format(args.n_epochs))
    log_file.write('learning_rate  = {} \n'.format(args.lr))
    log_file.write('random_seed    = {:d} \n'.format(args.random_seed))


    # start timer
    start_time = time.time()
    targ_img, style_img = load_imgs(args.targ_img_loc, args.targ_on, args.style_img_loc, args.style_on)
    log_file.write('images loaded successfully \n \n')

    end_time = time.time()
    time_taken = end_time - start_time

    #plt.figure(wt_index)
    #plt.subplots_adjust(hspace=0.5)
    #plt.subplot(211)
    #plt.title('Loss plot ' + type_ + ' distribution')
    #plt.xlabel('Epochs')
    #plt.ylabel('Training Loss')
    #plt.plot(loss_list)
    #plt.subplot(212)
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.plot(train_acc_list, 'r', label='training')
    #plt.plot(test_acc_list, 'b', label='testing')
    #plt.legend(loc='lower right')
    #plt.savefig('./plots/' + today_time + '_' + args.log_name + '_' + type_ + '.png')
    #plt.close('all')

    print('Time taken: {:.2f}mins \n'.format(time_taken/60))
    log_file.write('Time taken for : {:.2f}s \n'.format(time_taken))

    log_file.close()

    
