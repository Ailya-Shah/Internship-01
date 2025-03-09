!pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolo11n-cls.pt")
results = model.train(
    data="/content/Detect,-Count,-And-Visualize-Single-label-Classification-1/",
    epochs=100,
    imgsz=64,
    batch=16,
    workers=8,
    name="train5"
)
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model

dataset_path = "/content/Detect,-Count,-And-Visualize-Single-label-Classification-1"

# Perform validation with explicit dataset path
metrics = model.val(data=dataset_path, split="test") 
metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy
from ultralytics import YOLO
model = YOLO("yolo11n-cls.pt")
model = YOLO("/content/runs/classify/train5/weights/best.pt")

results = model("/content/WhatsApp Image 2025-03-08 at 06.47.28_b09144d7.jpg")
