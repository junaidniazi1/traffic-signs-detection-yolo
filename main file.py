

"""# Traffic Signs Detection

## Download Dataset From Kaggle
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("pkdarabi/cardetection")

print("Path to dataset files:", path)

"""# 🔹 Step 1: Import Libraries & Prepare Dataset Path

Before preprocessing, we need to import required libraries and set up dataset paths. Run this first:

👉 Run this, and check if it prints your dataset path correctly.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Path to dataset (downloaded via kagglehub)
dataset_path = path  # already set in your code

print("Dataset available at:", dataset_path)

"""#🔹 Step 2: Explore Dataset Structure

Before preprocessing, let’s see what’s inside the dataset folder (classes, images, etc.).

 👉 This will show whether your dataset is arranged in class folders (e.g., /Stop/, /SpeedLimit/, etc.) or if it has a different format.
"""

# Check folder structure
for root, dirs, files in os.walk(dataset_path):
    print("Root:", root)
    print("Dirs:", dirs[:5])   # show first 5 folders (classes)
    print("Files:", files[:5]) # show first 5 files
    break  # only check top-level

"""# 🔹 Step 3: Inspect Inside the car/ Folder
👉 Run this — it should print how many traffic sign categories exist and show a few class names.
"""

car_path = os.path.join(dataset_path, "car")

# List class folders inside "car"
class_dirs = os.listdir(car_path)
print("Number of classes:", len(class_dirs))
print("Example classes:", class_dirs[:10])

"""YOLO-style structure with train/, valid/, and test/ folders, plus a data.yaml file.
Inside each of these (train, valid, test), you’ll have two subfolders:

images/ → contains pictures

labels/ → contains YOLO-format text files with bounding box annotations

⚠️ This dataset is for object detection (car detection), not direct classification.
But since you want traffic sign classification with Transfer Learning, we’ll slightly adjust our preprocessing:

Ignore YOLO labels (bounding boxes) for now.

Use the images from train/, valid/, and test/ folders.

For classification, we’ll need to group them by class (sign type).

# 🔹 Step 4: Load Images for Classification
👉 Run this to see how many images we have and confirm the filenames.
"""

train_path = os.path.join(car_path, "train", "images")
valid_path = os.path.join(car_path, "valid", "images")
test_path  = os.path.join(car_path, "test", "images")

print("Train images:", len(os.listdir(train_path)))
print("Valid images:", len(os.listdir(valid_path)))
print("Test images:", len(os.listdir(test_path)))

# Show a few sample image names
print("Sample train files:", os.listdir(train_path)[:5])

"""Your dataset is YOLO object detection format, with 15 traffic sign classes (Stop, Speed Limits, Red/Green light, etc.).

#🔹 Step 5: Confirm Framework

-> Since your dataset is already YOLO-ready, the cleanest way is:

-> Use Ultralytics YOLOv8 (easy, transfer learning included)

-> Train directly on data.yaml

-> Use pretrained YOLOv8 small model (yolov8s.pt)

-> Fine-tune on your 15 traffic sign classes

-> Train using your train/valid/test split
"""

!pip install ultralytics

from ultralytics import YOLO

# Load pretrained YOLOv8 model (transfer learning)
model = YOLO("yolov8s.pt")  # you can also try yolov8m.pt or yolov8n.pt

# Train on your dataset
model.train(data=os.path.join(car_path, "data.yaml"), epochs=50, imgsz=640)

"""# 🔹 YOLO Evaluation Metrics
Example:::
          YOLO will automatically report:

mAP50 → mean Average Precision at IoU=0.5

mAP50-95 → mean AP averaged over IoU thresholds (0.5 to 0.95, COCO standard)

Precision → fraction of predicted boxes that are correct

Recall → fraction of ground-truth boxes detected
"""

from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # load best model
metrics = model.val()
print(metrics)   # prints precision, recall, mAP

"""# Load the model"""

!pip install ultralytics
from ultralytics import YOLO

# Load trained model
model = YOLO("/content/drive/MyDrive/deep learning/runs/detect/train/weights/best.pt")

"""# 🔹 Step 1: Upload Image"""

from google.colab import files

# Upload image
uploaded = files.upload()

"""# 🔹 Step 2: Run Prediction on Uploaded Image"""

import os

# Get the uploaded file name
image_path = list(uploaded.keys())[0]

# Run YOLO prediction
results = model.predict(source=image_path, conf=0.25, save=True)

# Show prediction
from IPython.display import Image as IPImage
pred_path = results[0].save_dir / os.path.basename(image_path)  # saved result
IPImage(filename=pred_path)

"""# 🔹 Step 3: Print Predicted Classes & Confidence Scores"""

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])  # class id
        conf = float(box.conf[0]) # confidence
        label = model.names[cls_id]
        print(f"{label}: {conf:.2f}")

"""# 🔹 SHOW IMAGE"""

from IPython.display import Image as IPImage
import os

# Join string paths correctly
pred_path = os.path.join(results[0].save_dir, os.path.basename(image_path))

# Show the prediction
IPImage(filename=pred_path)

"""# 🔹 Save The Model"""

# Save best model explicitly
model.save("best_model.pt")

"""# 🔹 script for video prediction:"""

from ultralytics import YOLO
from google.colab import files
import os

# ------------------------------
# 1. Upload Video
# ------------------------------
uploaded = files.upload()   # upload your mp4 file
video_path = list(uploaded.keys())[0]

print(f"✅ Uploaded video: {video_path}")

# ------------------------------
# 2. Load YOLO Model
# ------------------------------
model = YOLO("/content/drive/MyDrive/deep learning/runs/detect/train/weights/best.pt")   # replace with your trained weights

# ------------------------------
# 3. Run Predictions on Video
# ------------------------------
results = model.predict(
    source=video_path,   # input video
    conf=0.25,           # confidence threshold
    save=True            # save output video
)

print("✅ Processed video saved in:", results[0].save_dir)

# ------------------------------
# 4. Download Result Video
# ------------------------------
import shutil
output_folder = results[0].save_dir
for file in os.listdir(output_folder):
    if file.endswith(".mp4"):
        output_video = os.path.join(output_folder, file)
        shutil.copy(output_video, "predicted_video.mp4")

files.download("predicted_video.mp4")

