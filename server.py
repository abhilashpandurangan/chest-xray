import os
import pickle
import tornado.web
from tornado.ioloop import IOLoop
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets , models , transforms 
from torch.optim import lr_scheduler 
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 
import copy 
import cv2
import PIL
from PIL import Image
from torch.autograd import Variable
import torch
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
train_datatransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485 , 0.456 ,0.406] ,[0.229 , 0.224 , 0.225])])
test_datatrannsform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.485 ,0.456 ,0.406] ,[0.229 , 0.224 ,0.225])])

train_image_dataset = datasets.ImageFolder(root='data/train',transform=train_datatransform)
test_image_dataset = datasets.ImageFolder(root ='data/test', transform=test_datatrannsform)
class_names = train_image_dataset.classes
print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.load('model/best/model_resnet.h5')


def ranges(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)

def check_BW(img1):
    r = ranges(img1)
    x=0
    l=[]
    while x<r.shape[0]:
        if r[x][0] == r[x][1] == r[x][2]:
            l.append(0)
        else:
            l.append(1)
        x+=1
    #print(sum(l))
    if sum(l)!=0 :
        print('NOT BLACK & WHITE')
        return 1
    else:
        print('BLACK & WHITE')
        return 0

def predict(image): 
    image_tensor = test_datatrannsform(image).float()
    image_tensor = Variable(image_tensor,requires_grad=True)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.cuda()
  
def range(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


class BaseHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        # Get the file
        img = self.request.files.get('file')[0]
        img_body = BytesIO(img['body'])
        # Save the file
        with open(os.path.join('./static/uploads', img['filename']), 'wb') as f:
            image = Image.open(img_body)
            image = image.resize((224, 224))
            image.save(f, format='PNG')
            
    
        image = Image.open(img_body)
        if check_BW(image):
                print('WRONG INPUT IMAGE!!')
                self.render('predict.html', pred="WRONG INPUT IMAGE!!",description = "PLEASE ENTER CHEST X-RAY ONLY",file=img['filename'])
        else:
            im = Image.open(img_body)  #imagepath
            to_pil = transforms.ToPILImage()
            trans1 = transforms.ToTensor()

            im1 = to_pil(trans1(im))
            ans = predict(im1)
            outp = model(ans)
            _, preds = torch.max(outp,1)
            p = [class_names[x] for x in preds]
            print("Predicted label for given image is:", p) #prediction
        
            print(img['filename'])
            print(str(p[0]))
            if (str(p[0]) == "cardiomegaly"):
                descr = "Cardiomegaly is a medical condition in which the heart is enlarged."
            elif (str(p[0]) == "opacity"):
                descr = "Lung opacities are vague, fuzzy clouds of white in the darkness of the lungs"
            else:
                descr = "Chest is in normal condition"

            self.render('predict.html', pred=str(p[0]).capitalize(),description = descr,file=img['filename'])


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/?', BaseHandler),
            (r'/predict/?', UploadHandler)
        ]

        settings = {
            'template_path': os.path.join(os.path.dirname(__file__), 'templates'),
            'static_path': os.path.join(os.path.dirname(__file__), 'static'),
            'debug': True
        }

        super(Application, self).__init__(handlers, **settings)


def main():
    app = Application()
    print('Starting your application at port number 5000')
    app.listen(5000)
    IOLoop.instance().start()


if __name__ == '__main__':
    main()