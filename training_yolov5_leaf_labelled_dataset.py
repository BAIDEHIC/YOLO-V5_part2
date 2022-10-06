# -*- coding: utf-8 -*-
"""training_yolov5_mask_labelled_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Plz91PHWwf04bYt21mnWp7qBcDIvJ6J6

# Training YOLOv5 custom dataset with ease

Check the full explanation on medium:

https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone
# %cd yolov5
# %pip install -qr requirements.txt  # install

import torch
from yolov5 import utils
display = utils.notebook_init()

# Commented out IPython magic to ensure Python compatibility.
# %cd ..

import shutil
import os, sys

zip_file = "archive.zip"

if os.path.isfile(zip_file):
  shutil.unpack_archive(zip_file, "D")
else:
  print(zip_file + " not found")

import os, shutil, random

# preparing the folder structure

full_data_path = '/content/D/Victorytomato/'
extension_allowed = '.JPG'
split_percentage = 90 #splitting the data

images_path = 'D/images/'
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path) #image directory
    
labels_path = 'D/labels/'
if os.path.exists(labels_path):
    shutil.rmtree(labels_path)
os.mkdir(labels_path)#creation of labels directory
    
training_images_path = images_path + 'training/' #train images
validation_images_path = images_path + 'validation/' #validation images
training_labels_path = labels_path + 'training/' #train txt files
validation_labels_path = labels_path +'validation/' #validation txt files
    
os.mkdir(training_images_path)
os.mkdir(validation_images_path)
os.mkdir(training_labels_path)
os.mkdir(validation_labels_path)

files = [] #appending the names of files under labels directory

ext_len = len(extension_allowed)

for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            strip = file[0:len(file) - ext_len]      
            files.append(strip)

random.shuffle(files)

size = len(files)                   

split = int(split_percentage * size / 100)

print("copying training data")
for i in range(split):
    strip = files[i]
                         
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, training_images_path) 
                         
    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, training_labels_path) 

print("copying validation data")
for i in range(split, size):
    strip = files[i]
                         
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, validation_images_path) 
                         
    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, validation_labels_path) 

print("finished")

f = open("dataset.yaml", "a")

f.write("train: /content/D/images/training/\n")#appending the details to yaml file 
f.write("val: /content/D/images/validation/\n")
f.write("nc: 2\n")
f.write("names: ['with leaf', 'without leaf']\n")
f.close()

# Commented out IPython magic to ensure Python compatibility.
# %cd yolov5
!python train.py --img 640 --batch 16 --epochs 5 --data ../dataset.yaml --weights yolov5s.pt #training the model

#!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source ../test.jpg

#from IPython.display import display
#from PIL import Image

#image_path = "runs/detect/exp/test.jpg"
#display(Image.open(image_path))

