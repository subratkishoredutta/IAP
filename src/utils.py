

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import cv2
from torchvision import models
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import timm



def get_patch(image,delta,patch_X,patch_Y):
  patch_size = delta.shape[-1]
  x_p = image.clone()
  _, _, h, w = image.size()
  # Ensure the patch dimensions are within bounds
  patch_X = max(0, min(patch_X, h - patch_size))
  patch_Y = max(0, min(patch_Y, w - patch_size))
  temp = x_p[:,:,patch_X:patch_X+patch_size, patch_Y:patch_Y+patch_size]
  return temp

def apply_patch(image,delta,patch_X,patch_Y):
  patch_size = delta.shape[-1]
  x_p = image.clone()
  _, _, h, w = image.size()

  # Ensure the patch dimensions are within bounds
  patch_X = max(0, min(patch_X, h - patch_size))
  patch_Y = max(0, min(patch_Y, w - patch_size))
  x_p[:,:,patch_X:patch_X+patch_size, patch_Y:patch_Y+patch_size]=delta
  return x_p


## function to find the most busy patch in the image
def get_std_x(input_tensor,c=36):
  # Assuming you have an image with dimensions 224x290
  image = input_tensor.clone()
  image.requires_grad_()
  #plt.imshow(image.detach().numpy())
  #print(image.shape) 
  subset_size_x=3
  subset_size_y=1
  stride = 1
  stdten = torch.zeros(input_tensor.shape[2],input_tensor.shape[3])
  for i in range(0, image.size(2) - subset_size_y + 1, stride):
    for j in range(0, image.size(3) - subset_size_x + 1, stride):
      subset = image[0][:,i:i+subset_size_y, j:j+subset_size_x]
      subsetR=subset[0]
      subsetG=subset[1]
      subsetB=subset[2]
      std = (subsetR.std()+subsetG.std()+subsetB.std())/3
      stdten[i][j]=std
  #plt.figure()
  #plt.imshow(stdten.detach().numpy())
  return stdten
  
def get_std_y(input_tensor):
  # Assuming you have an image with dimensions 224x290
  image = input_tensor.clone()
  image.requires_grad_()
  #plt.imshow(image.detach().numpy())
  #print(image.shape)
  subset_size_x=1
  subset_size_y=3
  stride = 1
  stdten = torch.zeros(input_tensor.shape[2],input_tensor.shape[3])
  for i in range(0, image.size(2) - subset_size_y + 1, stride):
    for j in range(0, image.size(3) - subset_size_x + 1, stride):
      subset = image[0][:,i:i+subset_size_y, j:j+subset_size_x]
      subsetR=subset[0]
      subsetG=subset[1]
      subsetB=subset[2]
      std = (subsetR.std()+subsetG.std()+subsetB.std())/3
      stdten[i][j]=std
      #print(subsetB)
      #print(subsetB)
  #plt.figure()
  #plt.imshow(stdten.detach().numpy())
  return stdten
 
def get_final_std(image):
  stdx=get_std_x(image)
  stdy=get_std_y(image)
  min_tensor = torch.sqrt(torch.min(stdx, stdy))
  return min_tensor


## functions for getting class activation maps

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)#swin_b(pretrained=True)#swin_base_patch4_window7_224_in22k(pretrained=True,num_classes = 5)
model.eval()

def reshape_transform(tensor, height=6, width=8):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), 
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layer = [model.layers[-1].blocks[-1].norm2]
cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)
 ##gradcam

def give_grad_mask(image,target_index=None):
  #mod = models.resnet50(pretrained=True)
  #del mod.fc
  grayscale_cam = cam(input_tensor=image,
                    targets=target_index,
                    eigen_smooth=True,
                    aug_smooth=True)

  grayscale_cam = grayscale_cam[0, :]
  return torch.tensor(grayscale_cam)
  #plt.imshow(mask)

def find_location(image,mask,window_size=43):
    # Get the dimensions of the image
    height, width = mask.size()
    # Initialize variables to store the maximum sum and its location
    max_sum = 0
    max_sum_location = (0, 0)

    
    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = mask[i:i+window_size, j:j+window_size]
            window_sum = torch.sum(window).item()  # Convert to Python scalar

            if window_sum > max_sum:
                max_sum = window_sum
                max_sum_location = (i, j)

    return max_sum_location,image[:,:,max_sum_location[0]:max_sum_location[0]+window_size,max_sum_location[1]:max_sum_location[1]+window_size]

def getPert(image,window_size=43):
  stdimg=get_final_std(image)
  sens = 1/(stdimg+0.001)
  mask = give_grad_mask(image)
  perturbPrior=mask/(sens+0.001)
  location,subimage = find_location(image,mask,window_size)
  return location,subimage,sens[location[0]:location[0]+window_size,location[1]:location[1]+window_size]


def getPert_tar(image,target,window_size=43):
  stdimg=get_final_std(image)
  sens = 1/(stdimg+0.001)
  mask = give_grad_mask(image,target)
  perturbPrior=mask/(sens+0.001)
  location,subimage = find_location(image,mask,window_size)
  return location,subimage,sens[location[0]:location[0]+window_size,location[1]:location[1]+window_size]
