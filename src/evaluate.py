
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "defectguard_resnet18.pth"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Evaluating on: {device}")

# Apply the same preprocessing as validation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# Load only our locally trained, trusted checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

classes = checkpoint["classes"]

assert test_dataset.classes == classes, (
    "Dataset classes do not match the trained model."
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

actual = []
predicted = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.cpu().tolist())

# Accuracy
accuracy = accuracy_score(actual, predicted)

print("\nDEFECTGUARD AI — TEST RESULTS")
print(f"Test images: {len(test_dataset)}")
print(f"Test accuracy: {accuracy:.2%}")

report = classification_report(
    actual,
    predicted,
    target_names=classes,
    zero_division=0,
)

print("\nClassification Report:")
print(report)

# Save text report
with (RESULTS_DIR / "classification_report.txt").open(
    "w", encoding="utf-8"
) as file:
    file.write(f"Test accuracy: {accuracy:.4f}\n\n")
    file.write(report)

# Save machine-readable metrics
report_dict = classification_report(
    actual,
    predicted,
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

with (RESULTS_DIR / "test_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(
        {
            "test_accuracy": accuracy,
            "test_count": len(test_dataset),
            "classification_report": report_dict,
        },
        file,
        indent=2,
    )

# Confusion matrix
cm = confusion_matrix(
    actual,
    predicted,
    labels=list(range(len(classes))),
)

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(cm, cmap="Blues")

fig.colorbar(image, ax=ax)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("DefectGuard AI - Confusion Matrix")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2
            else "black",
        )

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print("\nResults saved to:", RESULTS_DIR)
