from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
model.train(data="D:/HT/thigiacmay/BaiTap/DemSoLuongXe/data.yaml", epochs=50, batch=16, imgsz=640, augment=True)