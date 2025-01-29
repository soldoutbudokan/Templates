import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ------------------------------------------------
# Dataset (same as before)
# ------------------------------------------------
class ArtworkDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.*'):
                    self.images.append((str(img_path), self.class_to_idx[class_dir.name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------------------------
# Simple CNN
# ------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 x 28 x 28
        )
        # Compute the flattened size:
        # After 3 pooling layers, each dimension is 224/(2^3) = 28
        # So feature map size is 128 * 28 * 28 = 128 * 784 = 100352
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ------------------------------------------------
# Training function (same as before)
# ------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device):
    print(f"Starting training on {device}")
    model.to(device)
    if isinstance(criterion, nn.Module):
        criterion.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(dim=1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {running_loss/len(train_loader):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}%")
        print(f"  Val   Loss: {val_loss/len(val_loader):.3f} | "
              f"Val   Acc: {100.*val_correct/val_total:.2f}%")

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = "kaggle_dataset/images/images"
    full_dataset = ArtworkDataset(data_dir, transform=transform)
    
    # Train/validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(full_dataset.classes)

    # Replace VisionTransformer with the SimpleCNN
    model = SimpleCNN(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device=device)
