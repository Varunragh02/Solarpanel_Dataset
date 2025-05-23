import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# === Step 1: Dataset Preparation ===

source_root = "D:/Guvi/Solarpanel_Dataset/samples"
target_root = "D:/Guvi/Solarpanel_Dataset"

images_train = Path(target_root) / "images/train"
images_val = Path(target_root) / "images/val"
labels_train = Path(target_root) / "labels/train"
labels_val = Path(target_root) / "labels/val"

for folder in [images_train, images_val, labels_train, labels_val]:
    folder.mkdir(parents=True, exist_ok=True)

class_map = {
    "Bird-Drop": 0,
    "Clean": 1,
    "Dusty": 2,
    "Electrical-Damage": 3,
    "Physical-Damage": 4,
    "Snow-Covered": 5
}

split_ratio = 0.8

print("Preparing dataset...")
for class_name, class_id in class_map.items():
    image_folder = Path(source_root) / class_name
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for file_list, img_dest, label_dest in [
        (train_files, images_train, labels_train),
        (val_files, images_val, labels_val),
    ]:
        for img_path in file_list:
            new_name = f"{class_name}_{img_path.name}"
            dest_img_path = img_dest / new_name
            shutil.copy(img_path, dest_img_path)

            # Dummy label: class_id x_center y_center width height (normalized)
            label_path = label_dest / (dest_img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("Dataset preparation complete.")

# === Step 2: Create dataset.yaml ===

yaml_path = Path(target_root) / "dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {target_root.replace(os.sep, '/')}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names:\n")
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(f"  {idx}: {name}\n")

print(f"dataset.yaml created at: {yaml_path}")

# === Step 3: Train YOLOv8 with Built-in Early Stopping ===

model = YOLO("yolov8n.pt")

model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    name="solar_panel_detector",
    exist_ok=True,
    patience=5  # Enables early stopping if no improvement for 5 epochs
)

print("Training complete.")
