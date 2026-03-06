import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# -----------------------------
# CUDA Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Parameters
# -----------------------------

DATASET_PATH = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# -----------------------------
# Load Dataset
# -----------------------------

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

print("Training Images:",train_size)
print("Validation Images:",val_size)

# -----------------------------
# CNN Model
# -----------------------------

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x


model = CNNModel().to(device)

# -----------------------------
# Loss + Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LR)

# -----------------------------
# Training History
# -----------------------------

train_acc_history = []
val_acc_history = []

train_loss_history = []
val_loss_history = []

# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    train_acc = correct/total
    train_loss = train_loss/len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs,labels)

            val_loss += loss.item()

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    val_acc = correct/total
    val_loss = val_loss/len(val_loader)

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Acc: {train_acc:.4f}  Train Loss: {train_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}  Val Loss: {val_loss:.4f}")
    print("--------------------------------")

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("model",exist_ok=True)

torch.save(model.state_dict(),"model/model.pth")

print("Model Saved!")

# -----------------------------
# Create Graphs
# -----------------------------

os.makedirs("results",exist_ok=True)

# Accuracy Graph

plt.figure()
plt.plot(train_acc_history,label="Train Accuracy")
plt.plot(val_acc_history,label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_graph.png")

# Loss Graph

plt.figure()
plt.plot(train_loss_history,label="Train Loss")
plt.plot(val_loss_history,label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss_graph.png")

print("Graphs Saved in results folder")