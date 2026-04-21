from ultralytics import YOLO
import os

def train_model():
    # 1. Load a pretrained YOLOv8 model (Nano is fastest for testing)
    model = YOLO('yolov8n.pt') 

    # 2. Train the model
    # data: Path to your data.yaml file
    # epochs: Number of training rounds
    # imgsz: Input image size (640 is standard)
    results = model.train(
        data='path/to/your/data.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=16,
        name='license_plate_detector'
    )
    
    print("Training complete. Model saved in runs/detect/license_plate_detector/weights/best.pt")

if __name__ == "__main__":
    train_model()
