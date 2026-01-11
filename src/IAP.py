# setting up the project page.
import os
os.chdir('../IAP') 
# importing the required packages
from utils import *
import torch
import torch.nn as nn
from torchvision.models import resnet50,vgg16
import json
from tqdm import tqdm
from PIL import Image
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import timm
from PIL import Image
import torchvision.transforms as transforms
import csv
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 4
random.seed(random_seed)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
])

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)#(pretrained=True)
model.eval()
print(device)
model = model.to(device)

with open("../IAP/src/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

patch_size=84
criterion = nn.CrossEntropyLoss()
criterion2  = nn.MSELoss()
learning_rates = [0.05,0.09,0.2] 
iterations = 1000
psnr = PSNR(data_range=1.0).to(device)
ssim = SSIM(data_range=1.0).to(device)
support = 1


target_class = 859 #toaster
datapath = "../IAP/imagenet1000main"
class_dir=sorted(os.listdir(datapath))




with open('../IAP/src/results/data/data_records_toaster.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['class','ssim_patches', 'ssim_images','psnr_patches','psnr_images','success','human distance','confidence'])
    for class_files in tqdm(class_dir):
      total_samples = 0
      success_rate = 0
      patch_coverage = 0
      ssim_monitor_patches=0
      ssim_monitor_images=0
      psnr_monitor_patches=0
      conf=0
      psnr_monitor_images=0
      class_name = imagenet_classes[int(class_files)]
      print(class_name)
      class_id = int(class_files)
      print(class_id)
      if class_id != []:
        file_list = os.listdir(os.path.join(datapath,class_files))
        random_file_list = random.sample(file_list, support+4)
        samidx=0
        for file in tqdm(random_file_list):
          image = Image.open(os.path.join(datapath,class_files,file))
          image = image.convert('RGB')
          image = preprocess(image)[None,:,:,:]
          image = image.to(device)
          if image.shape[1]==4:
            print("detected")
            continue
          if model(norm(image)).max(dim=1)[1].item()!=class_id:
            print(class_id)
            print(model(norm(image)).max(dim=1)[1].item())
            print("not classified")
          else:
            if samidx==support:
              break
            samidx+=1
            total_samples+=1
            patch_coverage+=100*patch_size**2/(image.shape[2]*image.shape[3])
            for learning_rate in learning_rates:
              (x,y),delta,sens = getPert_tar(image,None,patch_size)
              delta = delta.to(device)
              sens = sens.to(device)
              delta = torch.nn.Parameter(delta)
              temp=delta.clone()
              imagetem = image.clone()
              sen=sens.clone()
              for t in range(iterations) :
                x_p = apply_patch(image,delta,x,y)
                pred = model(norm(x_p))
                with torch.enable_grad():
                  loss = 1.5*criterion(pred,torch.LongTensor([target_class]).to(device)) - criterion(pred,torch.LongTensor([class_id]).to(device)) + 3*(sens.detach()*torch.abs(delta-temp)).sum()
                if pred.max(dim=1)[1].item() == target_class and nn.Softmax(dim=1)(pred)[0,pred.max(dim=1)[1].item()].item() >= 0.9:
                  break
                loss.backward()
                delta.data = delta.data - learning_rate*delta.grad.mean(dim=1)*delta.data*(1/sens)
                delta.grad.zero_()

              if pred.max(dim=1)[1].item() == target_class :
                conf=nn.Softmax(dim=1)(pred)[0,pred.max(dim=1)[1].item()].item()
                print("\nsuccess and max_prob:",conf)
                print(imagenet_classes[pred.max(dim=1)[1].item()])
                print("\nlr:", learning_rate)
                print("\n true class prob",nn.Softmax(dim=1)(pred)[0,class_id].item())
                success_rate+=1
                pur = apply_patch(image,delta,x,y)
                ssim_monitor_patches=ssim(temp, delta).item()
                ssim_monitor_images=ssim(pur, imagetem).item()
                psnr_monitor_patches=psnr(temp,delta).item()
                psnr_monitor_images=psnr(pur, imagetem).item()
                torch.save(imagetem, f"../IAP/src/results/imagetem/imagetem_{target_class}_{class_name}.pt") #../IAP/results/imagetem/
                torch.save(pur, f"../IAP/src/results/pur/pur_{target_class}_{class_name}.pt")
                torch.save(delta, f"../IAP/src/results/delta/delta_{target_class}_{class_name}.pt")
                torch.save(temp, f"../IAP/src/results/temp/temp_{target_class}_{class_name}.pt")
                break
              
              else:
                print("\nlr:", learning_rate)
                print("\nfailed and true class prob",nn.Softmax(dim=1)(pred)[0,class_id].item())
                print("\n target class prob",nn.Softmax(dim=1)(pred)[0,target_class].item())

            print(f"\nssim btw patches: {ssim(temp,delta).item()}")
            writer.writerow([class_name,ssim_monitor_patches, ssim_monitor_images,psnr_monitor_patches, psnr_monitor_images,success_rate,(sens.detach()*torch.abs(delta-temp)).sum().item(),conf])

