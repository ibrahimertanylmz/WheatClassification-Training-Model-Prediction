import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
import os
import shutil
import glob
from torch.utils.data import Dataset
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import tensorflow as tf
import sys
import h5py
import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def _get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print( _get_available_devices() )
print("Tensorlfow version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

class wheat_dataloader(Dataset):
  def __init__(self, root_dir=None, image_shape=(128,128)):
    self.class_dict = { 'bezostaya':0,
                        'dropi-torex':1,
                        'esperia':2,
                        'gerek':3,
                        'krasunia':4,
                        'kirac':5,
                        'maden':6,
                        'misiia':7,
                        'mufitbey':8,
                        'qality':9,
                        'rumeli':10,
                        'syrena':11,
                        'tosunbey':12,
                        'yubileynaus':13
                        }

    images_folder_path_1 = root_dir + '/*/*.JPG'
    images_folder_path_2 = root_dir + '/*/*.jpg'
    image_paths_1 = glob.glob(images_folder_path_1)
    image_paths_2 = glob.glob(images_folder_path_2)
    self.image_paths = image_paths_1 + image_paths_2

    self.transform = transforms.Compose(
                                        [transforms.ToTensor(),
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                        #transforms.Normalize((0.5), (0.5))]
                                       )

    self.image_shape = image_shape
  def __len__(self):
    return len(self.image_paths)
    
  def __getitem__(self, index):
    im_path = self.image_paths[index]
    label = self.class_dict[im_path.split('/')[-2]]
    #image = cv2.imread(im_path, 0)
    image = cv2.imread(im_path)
    image = cv2.resize(image, self.image_shape, interpolation = cv2.INTER_AREA)
    if self.transform is not None:
        image_tensor = self.transform(image)

    #image_tensor = torch.tensor(image).unsqueeze(0)
    label_tensor = torch.tensor(label)  
    return image_tensor, label_tensor

train_path = '/home/sm/data/yusuf/wheat-classification/wheat-dataset/wheat-train'
val_path = '/home/sm/data/yusuf/wheat-classification/wheat-dataset/wheat-validation'
image_shape=(256,256)
val_dataset = wheat_dataloader(val_path, image_shape=image_shape)
train_dataset = wheat_dataloader(train_path, image_shape=image_shape)



for i in range(1):
  rand_indx = random.randint(0, len(train_dataset))
  sample = train_dataset[rand_indx]
  print('label of this sample is: ', sample[1])
  plt.imshow(sample[0].numpy()[0])
  #plt.imshow(sample[0].numpy().transpose(1,2,0))
  plt.show()


batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1)
    self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)
    self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.dropout = nn.Dropout2d(0.05)
    self.batchnorm1 = nn.BatchNorm2d(16)
    self.batchnorm2 = nn.BatchNorm2d(32)
    self.batchnorm3 = nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(64*2*2,512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 14)
      
  def forward(self, x):
    x = self.batchnorm1(F.relu(self.conv1(x)))
    x = self.batchnorm2(F.relu(self.conv2(x)))
    x = self.dropout(self.batchnorm2(self.pool(x)))
    x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
    x = self.dropout(self.conv4(x))
    x = torch.flatten(x, 1)
    x = self.dropout(self.fc1(x))
    x = self.dropout(self.fc2(x))
    x = F.log_softmax(self.fc3(x),dim = 1)
    return x

from torch.utils.mobile_optimizer import optimize_for_mobile
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096,14)
model.load_state_dict(torch.load("/home/sm/data/yusuf/wheat-classification/model_alexnet6.pt"))
model.eval()

example = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/sm/data/yusuf/wheat-classification/model_alexnet6_scrOptoRAN.pth")



#example = torch.rand(1, 3, 256, 256)
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module_optimized = optimize_for_mobile(traced_script_module)
#traced_script_module_optimized._save_for_lite_interpreter("/home/sm/data/yusuf/wheat-classification/model_mobilenetv2Optimized.ptl")



