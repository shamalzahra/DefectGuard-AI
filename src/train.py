
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Configuration
SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Automatically use a GPU if the environment supports it
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Training on: {DEVICE}")

# Image preprocessing
# NEU images are grayscale, but pretrained ResNet18
# expects three-channel input.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

assert train_dataset.classes == val_dataset.classes

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace the final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    len(train_dataset.classes)
)

# Freeze the pretrained feature extractor
for parameter in model.parameters():
    parameter.requires_grad = False

# Train only the new classification layer
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

best_val_accuracy = 0.0
best_state = None
history = []

for epoch in range(EPOCHS):

    # Training
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    train_loss = total_loss / total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Acc: {val_accuracy:.2%}"
    )

    # Save best model according to validation accuracy
    if best_state is None or val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

# Save the best trained model
torch.save({
    "model_state_dict": best_state,
    "classes": train_dataset.classes,
    "architecture": "resnet18",
    "validation_accuracy": best_val_accuracy,
}, MODEL_DIR / "defectguard_resnet18.pth")

# Save training history
with (MODEL_DIR / "training_history.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(history, file, indent=2)

print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Model saved in: {MODEL_DIR}")
