
from pathlib import Path
from collections import Counter
import csv
import shutil

from sklearn.model_selection import train_test_split

# Project paths
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "NEU-DET"
OUTPUT = ROOT / "data" / "processed"

# Defect categories
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Find images in an existing dataset split
def collect_images(split_name):
    folder = SOURCE / split_name / "images"

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    records = []

    for class_name in CLASSES:
        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

        images = sorted(
            p for p in class_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not images:
            raise ValueError(
                f"No images in {class_folder}"
            )

        for image in images:
            records.append((image, class_name))

    return records


def main():
    print("Preparing DefectGuard AI dataset...")

    # Preserve the original validation images
    # as our final, untouched test set.
    original_train = collect_images("train")
    original_test = collect_images("validation")

    train_records, val_records = train_test_split(
        original_train,
        test_size=0.20,
        random_state=42,
        stratify=[label for _, label in original_train],
    )

    splits = {
        "train": train_records,
        "val": val_records,
        "test": original_test,
    }

    # Avoid overwriting an existing prepared dataset
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. "
            "Review it before preparing again."
        )

    OUTPUT.mkdir(parents=True)

    for split_name, records in splits.items():
        for source_path, class_name in records:
            target_folder = OUTPUT / split_name / class_name
            target_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_path,
                target_folder / source_path.name
            )

    # Save an auditable split manifest
    with (OUTPUT / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class", "source_file"])

        for split_name, records in splits.items():
            for source_path, class_name in records:
                writer.writerow([
                    split_name,
                    class_name,
                    str(source_path.relative_to(SOURCE)),
                ])

    print("\nDataset preparation completed!")

    for split_name, records in splits.items():
        counts = Counter(label for _, label in records)
        print(f"\n{split_name.upper()}: {len(records)} images")
        for class_name in CLASSES:
            print(f"  {class_name}: {counts[class_name]}")

    print(f"\nSaved dataset to: {OUTPUT}")


if __name__ == "__main__":
    main()
