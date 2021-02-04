import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

import argparse
import os


def arg_parse():
	"""
	Parse arguements to the detect module
	
	"""
	
	parser = argparse.ArgumentParser(description='Configurable Transfer Learning Module')
   
	parser.add_argument("--width", dest = 'width', help = 
						"width of the dataset images",
						default = 224)
	parser.add_argument("--height", dest = 'height', help = 
						"height of the dataset images",
						default = 224)

	parser.add_argument("--dir", dest = 'dir', help = 
						"directory name",
						default = "data", type = str)


	
	return parser.parse_args()
	
args = arg_parse()




data_dir = args.dir

width = args.width
height = args.height

test_transforms = transforms.Compose([transforms.Resize((width, height)),
                                      transforms.ToTensor(),
                                     ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('trained_model.pth')
model.eval()
model


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index  

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes



to_pil = transforms.ToPILImage()
images, labels, classes = get_random_images(4)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()



