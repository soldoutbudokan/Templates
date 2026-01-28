import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pathlib import Path
import kaggle

# ------------------------------------------------
# Kaggle setup — expects ~/.kaggle/kaggle.json to exist
# ------------------------------------------------
print("Downloading dataset...")
kaggle.api.authenticate()
owner_slug = "ikarus777"
dataset_slug = "best-artworks-of-all-time"
download_path = './kaggle_dataset'
os.makedirs(download_path, exist_ok=True)

kaggle.api.dataset_download_files(
    dataset=f"{owner_slug}/{dataset_slug}",
    path=download_path,
    unzip=True
)

##################################################
# 1. Custom Dataset Class
##################################################
class ArtworkDataset(Dataset):
    """
    Reads image files from subdirectories (one folder per artist).
    """
    def __init__(self, root_dir, artist_to_idx, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.artist_to_idx = artist_to_idx
        self.samples = []
        
        for artist_name, idx in self.artist_to_idx.items():
            dir_path = self.root_dir / artist_name
            if not dir_path.exists():
                continue
            for img_path in dir_path.glob("*.*"):
                self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


##################################################
# 2. Training Function
##################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_running_loss += v_loss.item() * val_inputs.size(0)
                val_preds = val_outputs.argmax(dim=1)
                val_correct += val_preds.eq(val_labels).sum().item()
                val_total += val_labels.size(0)
        
        val_loss = val_running_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs


##################################################
# 3. Main Workflow
##################################################
def main():
    # Select device first
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ------------------------------------------------
    # A. Find Artists with ≥200 images
    # ------------------------------------------------
    data_dir = "kaggle_dataset/images/images"
    root = Path(data_dir)
    
    # Gather all subdirectories
    all_artists = [d for d in root.iterdir() if d.is_dir()]
    
    # Count images
    artist_counts = {}
    for artist_path in all_artists:
        count = len(list(artist_path.glob("*.*")))
        artist_counts[artist_path.name] = count
    
    # Keep only artists with ≥200 images
    filtered_artists = {name: cnt for name, cnt in artist_counts.items() if cnt >= 200}
    print("Artists kept:", filtered_artists.keys())

    # Sort by descending count
    sorted_artists = sorted(filtered_artists.items(), key=lambda x: x[1], reverse=True)
    artist_names = [item[0] for item in sorted_artists]
    counts = [item[1] for item in sorted_artists]

    # Map artist -> index
    artist_to_idx = {name: i for i, name in enumerate(artist_names)}
    num_classes = len(artist_names)

    # ------------------------------------------------
    # B. Compute Class Weights
    # ------------------------------------------------
    total_paintings = sum(counts)
    class_weights = [total_paintings / (num_classes * c) for c in counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ------------------------------------------------
    # C. Transforms and Dataset
    # ------------------------------------------------
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = ArtworkDataset(data_dir, artist_to_idx, transform=transform_train)
    
    # Split into train and validation
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use simpler transforms on validation set
    val_dataset.dataset.transform = transform_val

    # ------------------------------------------------
    # D. Data Loaders
    # ------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ------------------------------------------------
    # E. Load Pretrained ResNet50
    # ------------------------------------------------
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features

    # Replace the final FC layer
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, num_classes)
    )

    # Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    # Move model to device
    model = model.to(device)

    # ------------------------------------------------
    # F. Setup Loss and Optimizer
    # ------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ------------------------------------------------
    # G. Train
    # ------------------------------------------------
    epochs = 10
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs
    )

    # ------------------------------------------------
    # H. Plot and Save Results
    # ------------------------------------------------
    os.makedirs("output_plots", exist_ok=True)

    plt.figure()
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss by Epoch")
    plt.savefig("output_plots/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(range(epochs), train_accs, label="Train Acc")
    plt.plot(range(epochs), val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy by Epoch")
    plt.savefig("output_plots/accuracy_curve.png")
    plt.close()

    print("Training complete. Plots saved in 'output_plots/'.")


##################################################
# 4. Entry Point
##################################################
if __name__ == "__main__":
    main()
