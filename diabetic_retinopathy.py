# DR Training Pipeline with Focal Loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Update this path to your local dataset location
DATA_DIR = "data/gaussian_filtered_images"

# Data Transforms

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Dataset & Stratified Split

dataset = ImageFolder(DATA_DIR)
indices = list(range(len(dataset)))
targets = dataset.targets

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(ImageFolder(DATA_DIR, transform=train_transform), train_idx)
test_dataset = Subset(ImageFolder(DATA_DIR, transform=test_transform), test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Class Weights

classes = np.unique([targets[i] for i in train_idx])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=[targets[i] for i in train_idx]
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# Model

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Focal Loss Definition

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Initialize Focal Loss with class weights
criterion = FocalLoss(alpha=class_weights, gamma=2)

# Optimizer + LR Scheduler

optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=2,
    factor=0.5
)

# Training Loop

epochs = 15
best_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    # LR logging
    prev_lr = optimizer.param_groups[0]['lr']
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != prev_lr:
        print(f"LR reduced from {prev_lr} to {new_lr}")

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_dr_resnet18_focal.pth")

print("Best Validation Accuracy:", best_acc)

# Detailed Classification Report

print(classification_report(y_true, y_pred, target_names=dataset.classes))

# Grad-CAM Visualization
# Simple Grad-CAM class for ResNet

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Register hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
        return heatmap

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# Pick a test image

idx = 10  # index in test_dataset
img_tensor, label = test_dataset[idx]  # img_tensor is normalized
input_tensor = img_tensor.unsqueeze(0).to(device)

# Target the last conv layer of ResNet18
target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# Generate Grad-CAM heatmap
cam = gradcam.generate(input_tensor)

# Denormalize image for visualization
img_np = img_tensor.cpu().permute(1,2,0).numpy()
img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
img_np = np.clip(img_np, 0, 1)

# Convert heatmap to RGB
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

# Overlay heatmap on image
overlay = 0.5 * img_np + 0.5 * heatmap
overlay = np.clip(overlay, 0, 1)

# Display
plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.title(f"True label: {test_dataset.dataset.classes[label]}")
plt.axis("off")
plt.show()

# Remove hooks to prevent memory leak
gradcam.remove_hooks()
