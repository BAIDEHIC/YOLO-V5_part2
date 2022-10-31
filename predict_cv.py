import torch
import pandas

import torch
import PIL
from PIL import Image 
from LeNet import LeNet
import numpy as np
import csv

from torchvision import transforms
from datetime import datetime
import time
import os
import cv2 
# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', '/content/yolov5/runs/train/exp/weights/best.pt')  # custom trained model

# Images
im = '/content/tomato.JPG'  # or file, Path, URL, PIL, OpenCV, numpy, list
# Inference
results = model(im)
# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.xyxy[0]  # im predictions (tensor)
df=results.pandas().xyxy[0]
df=df.sort_values(by=['confidence'],ascending=False) #getting the sorted dataframe
data_dict = df.to_dict()
print(data_dict)


#iterating in a nested dictionary and appending to list

l=[]
for k,v in data_dict.items():
  for keys,vals in v.items():
    if keys==0:
      l.append(vals)
print(l)


#cropping image
img = cv2.imread('/content/yolov5/runs/detect/exp/tomato.JPG')
cropped_image = img[j:h+j,i:w]

cv2.imwrite("tom.JPG", cropped_image) 

img_url="/content/tom.JPG"
ind_time = datetime.now()
file = os.path.splitext(img_url)
img = Image.open(img_url).resize((224, 224))


key_to_classname=["Tomato__BacterialSpot",
"Tomato__EarlyBlight",
"Tomato__healthy",
"Tomato__LateBlight",
"Tomato__LeafMold",
"Tomato__SeptoriaLeafspot",
"Tomato__SpidermitesTwo-spotted_spider_mite",
"Tomato__TargetSpot",
"Tomato__Tomato_Mosaicvirus",
"Tomato__Tomato_yellowleaf_curl_virus"]
#Keys for each classname
class_id_to_key=[0,1,2,3,4,5,6,7,8,9]

c1=time.time()
model=LeNet()#Calling torch model
input_data = torch.randn(1, 3, 32,32)
scripted_model = torch.jit.trace(model, input_data)
model=LeNet()#load CNN class
model.load_state_dict(torch.load('/content/model (1).pth'))#load model

my_preprocess = transforms.Compose(
    [
        transforms.Resize(32),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
c2=time.time()

img = my_preprocess(img)
img = np.expand_dims(img, 0)

begin = time.time()
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))#prediction of disease
end = time.time()
print(end-begin)
image = cv2.imread(img_url)
image=cv2.putText(img=image, text=key_to_classname[torch_class_key], org=(100, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,255,255), 
thickness=2)#APPENDING TEXT

cv2.imwrite("/content/with_text.png",image)#generating output