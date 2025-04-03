import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AIVideoDetector(nn.Module):
    def __init__(self):
        super(AIVideoDetector, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)
    
    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class VideoFrameDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_files[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
        
        cap.release()
        
        if len(frames) == 0:
            return None, None
        
        selected_frame = frames[0]
        label = 1 if "real" in self.video_files[idx] else 0
        return selected_frame, label

dataset_path = "dataset"
train_data = VideoFrameDataset(video_folder=f"{dataset_path}/train", transform=transform)
val_data = VideoFrameDataset(video_folder=f"{dataset_path}/val", transform=transform)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

model = AIVideoDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / len(train_data)
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / len(val_data)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), "ai_video_detector.pth")
    print("Model saved successfully.")

train_model(model, train_loader, val_loader, criterion, optimizer)
