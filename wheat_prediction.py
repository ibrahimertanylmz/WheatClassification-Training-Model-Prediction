import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import json

model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096,14)
model.eval()
model.load_state_dict(torch.load("/home/sm/data/yusuf/wheat-classification/model_alexnet6.pt"))

 
image = cv2.imread("/home/sm/data/yusuf/wheat-classification/wheat-dataset/wheat-validation/tosunbey/IMG_20211215_203449.jpg")
image = cv2.resize(image,(256,256) , interpolation = cv2.INTER_AREA)
 
transform = transforms.Compose(
                                        [transforms.ToTensor(),
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                        #transforms.Normalize((0.5), (0.5))]
                                       )
 
image = transform (image)
out = model(image.unsqueeze(0))
pred =  torch.max(out, dim=1)[1].item()

wheat_classes = [
            'bezostaya',
            'dropi-torex',
            'esperia',
            'gerek',
            'krasunia',
            'kirac',
            'maden',
            'misiia',
            'mufitbey',
            'qality',
            'rumeli',
            'syrena',
            'tosunbey',
            'yubileynaus'
]
 
pred =  torch.max(out, dim=1)[1].item()
 
results = { }
 
results["prediction_result"] = wheat_classes[pred]
 
for i in range(14): 
  results[wheat_classes[i]] = out[0][i].item()
 
pred_results_json = json.dumps(results)
print(pred_results_json)

