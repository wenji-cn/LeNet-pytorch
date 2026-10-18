import torch
import torch.nn as nn

class LeNet(nn.Module):
	def __init__(self,dim,num_classes):
		super().__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(in_channels=dim,out_channels=20,kernel_size=5,stride=1,padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2=nn.Sequential(
			nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1,padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.fc=nn.Sequential(
			nn.Linear(2450,500),
			nn.ReLU(),
			nn.Linear(500,num_classes)
		)

	def	forward(self,x):
		x=self.conv1(x)
		x=self.conv2(x)
		x=x.view(x.size(0),-1)
		output=self.fc(x)

		return output



