#ACCEPTED to commint


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim


batchsize = 5 #sesuaikan agar di c item tidak error
learningrate = 0.003
traindatapath = "images/train/"
testdatapath = "images/test/"
transimage = transforms.Compose([
transforms.Resize(28),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])
traindata= torchvision.datasets.ImageFolder(root=traindatapath,transform=transimage)
trainloader= data.DataLoader(traindata, batch_size = batchsize, shuffle=True, num_workers=2)
testdata= torchvision.datasets.ImageFolder(root=testdatapath,transform=transimage)
testloader = data.DataLoader(testdata, batch_size = batchsize, shuffle=True, num_workers=2)
kelas = ('diversifolia', 'lacunosa')


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





dataiter = iter(trainloader)
images, labels = dataiter.next()
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % kelas[labels[j]] for j in range(batchsize))) #range 2 berarti hanya 2 kelas




class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6,16 , 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader,0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i % 20 == 19:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
			running_loss = 0.0

print('selesai training')



PATH = './hoya_net.pth'
torch.save(model.state_dict(),PATH)
print('selesai menyimpan')


print("jumlah contoh latihan:", len(traindata))
print("jumlah contoh test:", len(testdata))


dataiter = iter(testloader)
images, labels = dataiter.next()
# print images
imshow(torchvision.utils.make_grid(images))
print('sebenarnya: ', ' '.join('%5s' % kelas[labels[j]] for j in range(batchsize)))

#harus ada ini agar c item tidak 0dim error
net = Net()
net.load_state_dict(torch.load(PATH))
#akhir  harus ada ini agar c item tidak 0dim error
outputs = net(images) #samakan dengan variabel di atas

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % kelas[predicted[j]]
                              for j in range(batchsize)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Akurasi keseluruhan: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()

		for i in range(2):
			label = labels[i]
			class_correct[label] += c[i].item() #agar tidak error data harus sangat banyak
			class_total[label] += 1


for i in range(2):
	print('Akurasi dari %5s : %2d %%' %(kelas[i], 100 * class_correct[i] / class_total[i]))



