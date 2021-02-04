import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
import os

from torchvision import datasets, models



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
	parser.add_argument("--model", dest = 'model', help = 
						"model name",
						default = "resnet50", type = str)

	parser.add_argument("--dir", dest = 'dir', help = 
						"directory name",
						default = "data", type = str)


	
	return parser.parse_args()
	
args = arg_parse()






width = args.width
height = args.height


# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = args.dir
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((width, height)),
                                       transforms.ToTensor(),])
    test_transforms = transforms.Compose([transforms.Resize((width, height)),
                                      transforms.ToTensor(),])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,  transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, .2)


if (args.model == "resnet50"):
	print("Loading resnet50 model")
	model = models.resnet50(pretrained=True)

	for param in model.parameters():
	    param.requires_grad = False

	model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

elif (args.model == "vgg16"):
	print("Loading vgg16 model")
	model = models.vgg16(pretrained=True)

	for param in model.features.parameters():
		param.requires_grad = False

	dataset_list = os.listdir(data_dir) # dir is your directory path
	number_files = len(dataset_list)

	# change the number of classes 
	model.classifier[6].out_features = int(number_files)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

else:
	print("Unknown model. Exiting...")
	'''Put Newer models here'''
	exit(1)

model.to(device)

epochs = 3
steps = 0
running_loss = 0
print_every = 1
train_losses, test_losses = [], []
accuracy_rate = []


for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print("epoch=", epoch, ". Train loss: ")
            print(str({running_loss/print_every})+".. Test loss: "+ str({test_loss/len(testloader)})+".. Test accuracy: "+ str({accuracy/len(testloader)}))
            accuracy_rate.append(accuracy/len(testloader))
            running_loss = 0
            model.train()
torch.save(model, 'trained_model.pth')


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


plt.plot(accuracy_rate, label="Accuracy curve")
plt.show()


