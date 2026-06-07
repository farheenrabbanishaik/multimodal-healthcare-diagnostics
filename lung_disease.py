# LUNG ABNORMALITY CLASSIFICATION MODULE

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset Path

LUNG_DATA_DIR = "data/COVID-19_Radiography_Dataset"

print("Lung Dataset Path:", LUNG_DATA_DIR)

# Transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Dataset

full_dataset = datasets.ImageFolder(root=LUNG_DATA_DIR, transform=transform)

class_names = full_dataset.classes
print("Classes:", class_names)

labels = [label for _, label in full_dataset.samples]
print("Class Distribution:", Counter(labels))

# Stratified Train-Test Split

indices = list(range(len(full_dataset)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Class Weights (FIX OPACITY BIAS)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class Weights:", class_weights)

# Load Pretrained ResNet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Loss & Optimizer

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels_batch.size(0)
        correct += (preds == labels_batch).sum().item()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss/total:.4f} "
          f"Accuracy: {correct/total:.4f}")

# Evaluation

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels_batch.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Lung Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Model

lung_model_path = "lung_resnet18_model.pth"
torch.save(model.state_dict(), lung_model_path)
print("Lung model saved at:", lung_model_path)

# Grad-CAM for Lung Images

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1),
                            size=input_tensor.shape[2:],
                            mode='bilinear',
                            align_corners=False)

        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

# Example Grad-CAM Visualization
sample_img, sample_label = test_dataset[10]
input_tensor = sample_img.unsqueeze(0).to(device)

target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate(input_tensor)

img_np = sample_img.permute(1,2,0).cpu().numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

overlay = 0.5 * img_np + 0.5 * heatmap
overlay = np.clip(overlay, 0, 1)

plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.title(f"True: {class_names[sample_label]}")
plt.axis("off")
plt.show()

# Confidence-Based Prediction

def predict_lung_with_confidence(model, image_tensor, classes, threshold=0.70):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    class_index = predicted_class.item()
    class_label = classes[class_index]
    confidence_value = confidence.item()

    if confidence_value < threshold:
        decision = "Refer to Radiologist"
    else:
        decision = "Prediction Accepted"

    return {
        "decision": decision,
        "predicted_label": class_label,
        "confidence": confidence_value
    }

result = predict_lung_with_confidence(model, input_tensor, class_names)

print("Decision:", result["decision"])
print("Predicted Label:", result["predicted_label"])
print("Confidence:", round(result["confidence"], 3))