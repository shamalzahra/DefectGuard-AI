
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "defectguard_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Only load checkpoints you created or trust.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, classes


def predict_image(image, model, classes):
    image = image.convert("RGB")

    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    probabilities = probabilities.cpu().numpy()

    best_index = int(probabilities.argmax())

    scores = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(classes)
    }

    return {
        "class": classes[best_index],
        "confidence": float(probabilities[best_index]),
        "scores": scores,
    }
