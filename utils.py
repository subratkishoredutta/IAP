

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import cv2
from torchvision import models
import numpy as np

## function to find the most busy patch in the image
def get_patch_opt(input_tensor,c=36):
  # Assuming you have an image with dimensions 224x290
  image = input_tensor.clone()
  image.requires_grad_()

  # Define Sobel filters for x and y gradients with three channels
  sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32).repeat(1, 3, 1, 1)
  sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32).repeat(1, 3, 1, 1)

  # Calculate gradients in x and y directions
  gradient_x = F.conv2d(image, sobel_x, stride=1, padding=1)
  gradient_y = F.conv2d(image, sobel_y, stride=1, padding=1)

  # Calculate the magnitude of the gradients for each pixel
  gradient_magnitude = torch.norm(gradient_x, dim=1) + torch.norm(gradient_y, dim=1)

  # Define the size of the sliding window
  patch_size = (c, c)

  # Initialize variables to store the maximum sum and corresponding coordinates
  max_sum = 0
  max_coordinates = (0, 0)

  # Loop over possible window positions and calculate the sum of gradients
  for i in range(gradient_magnitude.shape[1] - patch_size[0] + 1):
      for j in range(gradient_magnitude.shape[2] - patch_size[1] + 1):
          window_sum = torch.sum(gradient_magnitude[:, i:i+patch_size[0], j:j+patch_size[1]])
          if window_sum > max_sum:
              max_sum = window_sum
              max_coordinates = (i, j)

  # Define the size of the patch you want

  #print(max_coordinates)
  # Extract the patch with the highest sum of gradient magnitudes within the sliding window

  max_sum_patch = image[:, :,max_coordinates[0]:max_coordinates[0]+patch_size[0], max_coordinates[1]:max_coordinates[1]+patch_size[1]]

  return max_coordinates,max_sum_patch


# Now 'max_sum_patch' contains the patch with the highest sum of gradient magnitudes within the sliding window

def give_loc(delta):
  patch_size = delta.shape[-1]
  patch_X = torch.randint(0, 224 - patch_size, (1, ))
  patch_Y = torch.randint(0, 290 - patch_size, (1, ))

  return patch_X,patch_Y

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

def move_check(x,y,image,delta,stride=2,random=False):
  moves=[(stride,0),(-stride,0),(0,stride),(0,-stride)]#(+x,-x,+y,-y)
  maxloss = nn.CrossEntropyLoss()(model(norm(apply_patch(image,delta,x,y))),torch.LongTensor([304]))
  #print(maxloss)
  optim_move = (x,y)
  if random == True:
    moves=moves[torch.randint(0, 4, (1,)).item()]
  x_p = apply_patch(image,delta,x,y)
  for (move_x,move_y) in moves:
    #print(move_x,move_y)
    #print(nn.CrossEntropyLoss()(model(norm(apply_patch(image,delta,x+move_x,y+move_y))),torch.LongTensor([304])))
    if nn.CrossEntropyLoss()(model(norm(apply_patch(image,delta,x+move_x,y+move_y))),torch.LongTensor([304]))<maxloss:
      optim_move = (move_x,move_y)
  if optim_move!=0:
    x_p=apply_patch(image,delta,x+optim_move[0],y+optim_move[1])

  return x_p,x+optim_move[0],y+optim_move[1]

## patch refining phase
def refine(image,delta,temp,patch_X,patch_Y,epsilon=0.01):
  pred = model(norm(apply_patch(image,delta,patch_X,patch_Y)))
  refined_patch = delta.clone().detach()
  for i in range(delta.shape[2]):
    for j in range(delta.shape[-3]):
      tempDelta = delta.clone()
      tempDelta[:,:,i,j] = temp[:,:,i,j]
      tempimg = apply_patch(image,tempDelta,patch_X,patch_Y)
      tempred = model(norm(tempimg))
      if model(norm(apply_patch(image,refined_patch,patch_X,patch_Y))).max(dim=1)[1].item() != pred.max(dim=1)[1].item():#loss -criterion(tempred,torch.LongTensor([class_ind])) < -epsilon: ## change the condition to same class prediction
        return final
      else:
        final = refined_patch.clone()
        refined_patch[:,:,i,j]= temp[:,:,i,j]
  return final

def find_keys(dictionary, target_value):
    keys = [key for key, value in dictionary.items() if value.lower() == target_value.lower()]
    return keys

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

model=models.resnet50(pretrained=True)

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)

            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		#print('classfier=',output.size())
		if self.cuda:
			output = output.cpu()
			output = model.fc(output).cpu()
		else:
			output = model.fc(output)
		return target_activations, output

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda='cpu'):
		self.model = model
		self.model.eval()
		self.device = use_cuda
		if self.device:
			self.model = model.to(self.device)

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.device:
			features, output = self.extractor(input.to(self.device))
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.device:
			one_hot = torch.sum(one_hot.to(self.device) * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if True:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)
	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		#print(input.grad)
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		#self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def give_grad_mask(image,target_index=None):
  mod = models.resnet50(pretrained=True)
  del mod.fc
  grad_cam = GradCam(mod ,target_layer_names = ["layer4"])
  mask = grad_cam(image, target_index) 
  return torch.tensor(mask)
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



def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg
def sliding_window(image: np.ndarray, stepSize: int, windowSize: int):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])


def uiq(
    org_img: np.ndarray, pred_img: np.ndarray, step_size: int = 1, window_size: int = 8
) -> float:
    """
    Universal Image Quality index
    """
    _assert_image_shapes_equal(org_img, pred_img, "UIQ")

    org_img = org_img.astype(np.float32)
    pred_img = pred_img.astype(np.float32)
    q_all = []
    for (x, y, window_org), (x, y, window_pred) in zip(
        sliding_window(
            org_img, stepSize=step_size, windowSize=(window_size, window_size)
        ),
        sliding_window(
            pred_img, stepSize=step_size, windowSize=(window_size, window_size)
        ),
    ):
        # if the window does not meet our desired window size, ignore it
        if window_org.shape[0] != window_size or window_org.shape[1] != window_size:
            continue
        
        # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
        if org_img.ndim == 2:
            org_img = np.expand_dims(org_img, axis=-1)

        org_band = window_org.transpose(2, 0, 1).reshape(-1, window_size ** 2)
        pred_band = window_pred.transpose(2, 0, 1).reshape(-1, window_size ** 2)
        org_band_mean = np.mean(org_band, axis=1, keepdims=True)
        pred_band_mean = np.mean(pred_band, axis=1, keepdims=True)
        org_band_variance = np.var(org_band, axis=1, keepdims=True)
        pred_band_variance = np.var(pred_band, axis=1, keepdims=True)
        org_pred_band_variance = np.mean(
            (org_band - org_band_mean) * (pred_band - pred_band_mean), axis=1, keepdims=True
        )

        numerator = 4 * org_pred_band_variance * org_band_mean * pred_band_mean
        denominator = (org_band_variance + pred_band_variance) * (
            org_band_mean**2 + pred_band_mean**2
        )

        q = np.nan_to_num(numerator / denominator)
        q_all.extend(q.tolist())

    if not q_all:
        raise ValueError(
            f"Window size ({window_size}) is too big for image with shape "
            f"{org_img.shape[0:2]}, please use a smaller window size."
        )

    return np.mean(q_all)



def sam(org_img: np.ndarray, pred_img: np.ndarray, convert_to_degree: bool = True) -> float:
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
    _assert_image_shapes_equal(org_img, pred_img, "SAM")

    numerator = np.sum(np.multiply(pred_img, org_img), axis=-1)
    denominator = np.linalg.norm(org_img, axis=-1) * np.linalg.norm(pred_img, axis=-1)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = np.rad2deg(sam_angles)

    return np.nan_to_num(np.mean(sam_angles))


def sre(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Signal to Reconstruction Error Ratio
    """
    _assert_image_shapes_equal(org_img, pred_img, "SRE")

    org_img = org_img.astype(np.float32)
    
    # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)

    sre_final = []
    for i in range(org_img.shape[2]):
        numerator = np.square(np.mean(org_img[:, :, i]))
        denominator = (np.linalg.norm(org_img[:, :, i] - pred_img[:, :, i])) / (
            org_img.shape[0] * org_img.shape[1]
        )
        sre_final.append(numerator / denominator)

    return 10 * np.log10(np.mean(sre_final))


