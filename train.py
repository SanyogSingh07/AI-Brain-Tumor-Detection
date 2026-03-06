import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset_path = r"C:\Users\sanyo\OneDrive\Desktop\brain_tumor_ai\dataset"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(dataset_path, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])

train_loader = DataLoader(train_data,batch_size=16,shuffle=True)
val_loader = DataLoader(val_data,batch_size=16)
test_loader = DataLoader(test_data,batch_size=16)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*26*26,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x

model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 10

train_losses=[]
val_losses=[]
train_acc=[]
val_acc=[]

for epoch in range(epochs):

    model.train()
    running_loss=0
    correct=0
    total=0

    for images,labels in train_loader:

        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        _,pred=torch.max(outputs,1)

        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

    train_losses.append(running_loss/len(train_loader))
    train_acc.append(correct/total)

    model.eval()

    correct=0
    total=0
    vloss=0

    with torch.no_grad():

        for images,labels in val_loader:

            images,labels=images.to(device),labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            vloss+=loss.item()

            _,pred=torch.max(outputs,1)

            correct+=(pred==labels).sum().item()
            total+=labels.size(0)

    val_losses.append(vloss/len(val_loader))
    val_acc.append(correct/total)

    print(f"Epoch {epoch+1}/{epochs} Train Acc:{train_acc[-1]:.2f} Val Acc:{val_acc[-1]:.2f}")

os.makedirs("model",exist_ok=True)
torch.save(model.state_dict(),"model/model.pth")

os.makedirs("results",exist_ok=True)

plt.plot(train_acc,label="train")
plt.plot(val_acc,label="val")
plt.legend()
plt.title("Accuracy")
plt.savefig("results/accuracy.png")
plt.close()

plt.plot(train_losses,label="train")
plt.plot(val_losses,label="val")
plt.legend()
plt.title("Loss")
plt.savefig("results/loss.png")
plt.close()

y_true=[]
y_pred=[]

with torch.no_grad():

    for images,labels in test_loader:

        images=images.to(device)

        outputs=model(images)

        _,pred=torch.max(outputs,1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

cm=confusion_matrix(y_true,y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

print("Training complete")
