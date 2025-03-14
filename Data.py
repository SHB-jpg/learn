import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from cv2.gapi import kernel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")
file_path="D:\\mechinelearning_data\\chinesenumber\\chinese_mnist.csv"
class ChineseDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples=[]
        for idx in range(len(self.img_labels)):
            img_path=os.path.join(self.img_dir,f"input_{self.img_labels.iloc[idx,0]}_{self.img_labels.iloc[idx,1]}_{self.img_labels.iloc[idx,2]}.jpg")
            labels=self.img_labels.loc[idx,"code"]-1
            self.samples.append((img_path,labels))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path,labels=self.samples[idx]
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found!")
        img=Image.open(img_path).convert("RGB")
        if self.transform:
            img=self.transform(img)
        if self.target_transform:
            labels=self.target_transform(labels)
        return img,labels

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加BN
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加BN
            nn.ReLU()
        )
        self.shortcut=nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),  # 添加BN
            )
    def forward(self,x):
        residual=x
        x=self.features(x)
        x=x+self.shortcut(residual)
        x=nn.ReLU(inplace=False)(x)
        return x
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),  # 添加BN
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.layer1=self._make_layer(64,2,stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 15)
    def _make_layer(self, out_channels, num_blocks, stride):
        layers=[]
        layers.append(ResidualBlock(self.in_channels, out_channels,stride=1))
        self.in_channels=out_channels
        for i in range(1,num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels,stride))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

if __name__=="__main__":
    data_augmentation=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    dataset=ChineseDataset(
        file_path,
        "D:\\mechinelearning_data\\chinesenumber\\data\\data",
        transform=data_augmentation
    )
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    data_loader=DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type=="cuda" else False,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    model=ResNet().to(device)
    print(next(model.parameters()).device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001)

    for i in range(50):
        model.train()
        total_loss=0.0
        for img, labels in data_loader:
            img=img.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            optimizer.zero_grad()
            outputs=model(img)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(data_loader)
        print(f'Epoch [{i + 1}/50], Loss: {avg_loss:.4f}')

    # 计算验证集准确率
    model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in val_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
        print(f"Val Accuracy: {100 * correct / len(val_dataset):.2f}%")
torch.save(model.state_dict(), "chinese_number_model.pth")



