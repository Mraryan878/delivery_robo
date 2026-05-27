import cv2
import numpy as np
from ultralytics import YOLO

class AIModel:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize AI model
        
        Args:
            model_name: YOLO model name or path
            conf_threshold: Confidence threshold for detections
        """
        print(f"[INFO] Loading AI model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        print(f"[INFO] Model loaded successfully!")
    
    def detect(self, image):
        """
        Run detection on image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detections, each containing x_center, y_center, width, height, confidence, class_id
        """
        if image is None:
            return []
        
        # Resize for faster processing (optional)
        # image = cv2.resize(image, (320, 240))
        
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detections.append({
                    'x_center': x_center,
                    'y_center': y_center,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def detect_and_draw(self, image):
        """
        Run detection and draw bounding boxes on image
        
        Returns:
            Annotated image, list of detections
        """
        detections = self.detect(image)
        
        # Draw bounding boxes
        for det in detections:
            x1 = int(det['x1'])
            y1 = int(det['y1'])
            x2 = int(det['x2'])
            y2 = int(det['y2'])
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image, detections

# For testing the model independently
if __name__ == '__main__':
    # Test the model
    print("Testing AI Model...")
    model = AIModel()
    
    # Create a test image (black)
    test_image = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = model.detect(test_image)
    print(f"Test detections: {len(detections)}")