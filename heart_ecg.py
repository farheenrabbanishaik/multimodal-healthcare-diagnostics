# IMPORTS & DEVICE

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOAD DATASET

heart_path = "data/heart"

# Dataset:
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat

# Optional: Download dataset
# !kaggle datasets download -d shayanfazeli/heartbeat -p data/heart
# !unzip -o data/heart/heartbeat.zip -d data/heart

train_df = pd.read_csv(f"{heart_path}/mitbih_train.csv", header=None)
test_df = pd.read_csv(f"{heart_path}/mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# NORMALIZATION

X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)

# FORMAT FOR CNN

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# WEIGHTED SAMPLER (ONLY THIS)

class_counts = np.bincount(y_train)
weights = 1. / class_counts
sample_weights = weights[y_train]

sample_weights = torch.tensor(sample_weights, dtype=torch.float)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# LABEL MAP

label_map = {
    0: 'Normal',
    1: 'Supraventricular (S)',
    2: 'Ventricular (V)',
    3: 'Fusion (F)',
    4: 'Unknown (Q)'
}

# FOCAL LOSS (NO CLASS WEIGHTS)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# MODEL

class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 23, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# TRAINING

model = ECG_CNN().to(device)

criterion = FocalLoss(gamma=2)   # 🔥 no class weights
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 25

print("Training started...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
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

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Accuracy={100*correct/total:.2f}%")

print("Training completed.")

# EVALUATION

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))
print("\nMacro F1 Score:", f1_score(all_labels, all_preds, average='macro'))

# SAVE MODEL

save_path = "models/heart_ecg_model.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(model.state_dict(), save_path)
print("Model saved at:", save_path)

# LOAD MODEL

model_path = "models/heart_ecg_model.pth"

model = ECG_CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully!")

# LOAD ONE SAMPLE FROM TEST SET
# Re-load dataset (only small part needed)

test_df = pd.read_csv("data/heart/mitbih_test.csv", header=None)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Normalize (IMPORTANT same as training)
X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)

X_test = np.expand_dims(X_test, axis=1)

# Pick one sample
idx = 0
input_sample = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(device)
true_label = int(y_test[idx])

# PREDICTION

output = model(input_sample)
pred_class = torch.argmax(output, dim=1).item()

print("True Label:", label_map[true_label])
print("Predicted Label:", label_map[pred_class])

# INTERPRETABILITY USING INTEGRATED GRADIENTS

ig = IntegratedGradients(model)

baseline = torch.zeros_like(input_sample).to(device)

attributions, delta = ig.attribute(
    input_sample,
    baselines=baseline,
    target=pred_class,
    n_steps=100,
    return_convergence_delta=True
)

print("Convergence Delta:", delta.item())

# PLOT RESULTS

signal = input_sample.detach().cpu().numpy()[0][0]
attributions = attributions.detach().cpu().numpy()[0][0]

# Normalize attribution
attributions = attributions / (np.max(np.abs(attributions)) + 1e-8)

plt.figure(figsize=(12,4))
plt.plot(signal, label="ECG Signal")
plt.plot(attributions, label="Integrated Gradients", alpha=0.7)
plt.legend()
plt.title("ECG Signal with Important Regions Highlighted")
plt.show()