from ultralytics import YOLO
import cv2
import os

def run_inference(image_path, model_path='runs/detect/license_plate_detector/weights/best.pt'):
    # 1. Load your custom trained model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you run train.py?")
        return

    model = YOLO(model_path)

    # 2. Run detection
    results = model.predict(source=image_path, conf=0.25, save=True)

    # 3. Process and display results
    for result in results:
        # Show the image with bounding boxes
        result.show()
        
        # Print detected coordinates for debugging
        print(f"Detected {len(result.boxes)} plates in {image_path}")
        for box in result.boxes:
            print(f"Box: {box.xyxy} Confidence: {box.conf}")

if __name__ == "__main__":
    # Test on a single image
    test_img = "test_data/car_sample.jpg" 
    run_inference(test_img)
