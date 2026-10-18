import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from LeNet import LeNet

def get_dataloaders(batch_size=64):
	transform=transforms.Compose([
		#transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,),(0.3081,)),
		])
	train_set=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
	test_set=datasets.MNIST(root="./data",train=False,download=True,transform=transform)
	
	train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
	test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

	return train_loader,test_loader



def train_one_epoch(model,loader,optimizer,criterion,device):
	model.train()
	running_loss,correct,total=0.0,0,0

	for images,labels in loader:
		images,labels=images.to(device),labels.to(device)
		
		optimizer.zero_grad()
		outputs=model(images)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss+=loss.item()*images.size(0)
		_,preds=outputs.max(1)
		correct+=(preds==labels).sum().item()
		total+=labels.size(0)
	
	return running_loss/total,correct/total



@torch.no_grad()
def evaluate(model,loader,criterion,device):
	model.eval()
	running_loss,correct,total=0.0,0,0

	for images,labels in loader:
		images,labels=images.to(device),labels.to(device)
		outputs=model(images)
		loss=criterion(outputs,labels)

		running_loss+=loss.item()*images.size(0)
		_,preds=outputs.max(1)
		correct+=(preds==labels).sum().item()
		total+=labels.size(0)

	return running_loss/total,correct/total



def main():
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device:{device}")

	train_loader,test_loader=get_dataloaders(batch_size=64)

	model=LeNet(dim=1,num_classes=10).to(device)
	criterion=nn.CrossEntropyLoss()
	optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

	epochs=10
	for epoch in range(1,epochs+1):
		train_loss,train_acc=train_one_epoch(model,train_loader,optimizer,criterion,device)
		test_loss,test_acc=evaluate(model,test_loader,criterion,device)

		print(f"Epoch [{epoch}/{epochs}] "
			  f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
			  f"Test Loss:{test_loss:.4f} Acc:{test_acc:.4f}")

	torch.save(model.state_dict(),"./pth_data/lenet.pth")
	print("Model saved to lenet.pth")



if __name__=="__main__":
	main()












