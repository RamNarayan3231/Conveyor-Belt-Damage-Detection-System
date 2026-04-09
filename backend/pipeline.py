# #!/usr/bin/env python3
# """
# Conveyor Belt Damage Detection Pipeline
# Command line interface for batch processing
# """

# import argparse
# import json
# import cv2
# import numpy as np
# from pathlib import Path
# from typing import Dict, Any
# import sys

# class DamageDetector:
#     """Damage detector using computer vision"""
    
#     def detect_damage(self, image_path: str) -> Dict[str, Any]:
#         """Detect damage in an image"""
#         # Load image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Cannot read image: {image_path}")
        
#         # Simple detection using edge detection
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
        
#         # Find contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         detections = {}
#         for i, contour in enumerate(contours):
#             x, y, w, h = cv2.boundingRect(contour)
#             area = w * h
            
#             # Filter small detections
#             if area > 500:
#                 detections[str(i + 1)] = {
#                     "bbox_coordinates": [int(x), int(y), int(x + w), int(y + h)],
#                     "class": "damage",
#                     "confidence": min(0.95, 0.5 + area / 5000)
#                 }
        
#         # Draw detections
#         annotated = image.copy()
#         for det in detections.values():
#             x1, y1, x2, y2 = det['bbox_coordinates']
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(annotated, f"Damage: {det['confidence']:.0%}", 
#                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
#         return annotated, detections

# def main():
#     parser = argparse.ArgumentParser(description='Conveyor Belt Damage Detection Pipeline')
#     parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
#     parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
#     args = parser.parse_args()
    
#     # Create output directory
#     output_path = Path(args.output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # Find images
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
#     image_files = [f for f in Path(args.image_dir).iterdir() 
#                   if f.suffix.lower() in image_extensions]
    
#     print(f"Found {len(image_files)} images")
    
#     detector = DamageDetector()
    
#     for img_file in image_files:
#         print(f"Processing: {img_file.name}")
        
#         try:
#             annotated, detections = detector.detect_damage(str(img_file))
            
#             # Save annotated image
#             output_img = output_path / f"{img_file.stem}_annotated.jpg"
#             cv2.imwrite(str(output_img), annotated)
            
#             # Save JSON
#             output_json = output_path / f"{img_file.stem}.json"
#             with open(output_json, 'w') as f:
#                 json.dump(detections, f, indent=2)
            
#             print(f"  ✅ Saved: {output_img.name} and {output_json.name}")
            
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
    
#     print("\n✅ Pipeline completed!")

# if __name__ == "__main__":
#     main()


# # import os
# # import json
# # import argparse
# # from pathlib import Path
# # from typing import Dict, List, Tuple, Any, Optional
# # import cv2
# # import numpy as np
# # from ultralytics import YOLO


# # class ConveyorBeltDamageDetector:
# #     """Conveyor belt damage detection pipeline for scratches and edge damage."""
    
# #     def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
# #         """
# #         Initialize the detector.
        
# #         Args:
# #             model_path: Path to trained YOLO model weights
# #             conf_threshold: Confidence threshold for detections
# #             iou_threshold: IoU threshold for NMS
# #         """
# #         self.model_path = Path(model_path)
# #         self.conf_threshold = conf_threshold
# #         self.iou_threshold = iou_threshold
# #         self.model = None
# #         self.class_names = {0: 'scratch', 1: 'edge_damage'}
# #         self.colors = {
# #             'scratch': (0, 255, 255),      # Yellow for scratches
# #             'edge_damage': (0, 0, 255)      # Red for edge damage
# #         }
# #         self._load_model()
    
# #     def _load_model(self):
# #         """Load the YOLO model."""
# #         if self.model_path.exists():
# #             self.model = YOLO(str(self.model_path))
# #             print(f"Model loaded from {self.model_path}")
# #         else:
# #             raise FileNotFoundError(f"Model not found at {self.model_path}")
    
# #     def detect(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
# #         """
# #         Run detection on a single image.
        
# #         Args:
# #             image_path: Path to input image
            
# #         Returns:
# #             Tuple of (annotated_image, detections_dict)
# #         """
# #         # Read image
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             raise ValueError(f"Cannot read image: {image_path}")
        
# #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         height, width = image.shape[:2]
        
# #         # Run inference
# #         results = self.model(image_rgb, conf=self.conf_threshold, iou=self.iou_threshold)
        
# #         # Parse detections
# #         detections = {}
# #         annotated_image = image_rgb.copy()
        
# #         for result in results[0].boxes:
# #             if result is None:
# #                 continue
                
# #             # Get bounding box coordinates (x1, y1, x2, y2)
# #             x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
# #             confidence = float(result.conf[0])
# #             class_id = int(result.cls[0])
# #             class_name = self.class_names.get(class_id, 'unknown')
            
# #             # Store detection
# #             detection_id = len(detections) + 1
# #             detections[str(detection_id)] = {
# #                 "bbox_coordinates": [x1, y1, x2, y2],
# #                 "class": class_name,
# #                 "confidence": confidence
# #             }
            
# #             # Draw bounding box on image
# #             color = self.colors.get(class_name, (0, 255, 0))
# #             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
# #             # Add label
# #             label = f"{class_name}: {confidence:.2f}"
# #             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
# #             cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 5), 
# #                          (x1 + label_size[0], y1), color, -1)
# #             cv2.putText(annotated_image, label, (x1, y1 - 5),
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
# #         return annotated_image, detections
    
# #     def detect_from_array(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
# #         """
# #         Run detection on image array.
        
# #         Args:
# #             image: numpy array (BGR format)
            
# #         Returns:
# #             Tuple of (annotated_image, detections_dict)
# #         """
# #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# #         # Run inference
# #         results = self.model(image_rgb, conf=self.conf_threshold, iou=self.iou_threshold)
        
# #         # Parse detections
# #         detections = {}
# #         annotated_image = image_rgb.copy()
        
# #         for result in results[0].boxes:
# #             if result is None:
# #                 continue
                
# #             x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
# #             confidence = float(result.conf[0])
# #             class_id = int(result.cls[0])
# #             class_name = self.class_names.get(class_id, 'unknown')
            
# #             detection_id = len(detections) + 1
# #             detections[str(detection_id)] = {
# #                 "bbox_coordinates": [x1, y1, x2, y2],
# #                 "class": class_name,
# #                 "confidence": confidence
# #             }
            
# #             color = self.colors.get(class_name, (0, 255, 0))
# #             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
# #             label = f"{class_name}: {confidence:.2f}"
# #             cv2.putText(annotated_image, label, (x1, y1 - 5),
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
# #         return annotated_image, detections
    
# #     def process_directory(self, image_dir: str, output_dir: str) -> None:
# #         """
# #         Process all images in a directory.
        
# #         Args:
# #             image_dir: Directory containing input images
# #             output_dir: Directory to save outputs
# #         """
# #         # Create output directory
# #         output_path = Path(output_dir)
# #         output_path.mkdir(parents=True, exist_ok=True)
        
# #         # Supported image extensions
# #         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
# #         # Process each image
# #         image_files = [f for f in Path(image_dir).iterdir() 
# #                       if f.suffix.lower() in image_extensions]
        
# #         print(f"Found {len(image_files)} images to process")
        
# #         for img_file in image_files:
# #             print(f"Processing: {img_file.name}")
            
# #             try:
# #                 # Run detection
# #                 annotated_image, detections = self.detect(str(img_file))
                
# #                 # Save annotated image
# #                 output_image_path = output_path / f"{img_file.stem}.jpg"
# #                 cv2.imwrite(str(output_image_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                
# #                 # Save detections JSON
# #                 output_json_path = output_path / f"{img_file.stem}.json"
# #                 with open(output_json_path, 'w') as f:
# #                     json.dump(detections, f, indent=2)
                
# #                 print(f"  -> Saved: {output_image_path.name} and {output_json_path.name}")
                
# #             except Exception as e:
# #                 print(f"  -> Error processing {img_file.name}: {str(e)}")


# # def main():
# #     parser = argparse.ArgumentParser(description='Conveyor Belt Damage Detection Pipeline')
# #     parser.add_argument('--image_dir', type=str, required=True,
# #                        help='Path to directory containing input images')
# #     parser.add_argument('--output_dir', type=str, required=True,
# #                        help='Path to directory for output files')
# #     parser.add_argument('--model_path', type=str, default='backend/models/best.pt',
# #                        help='Path to trained model weights')
# #     parser.add_argument('--conf_threshold', type=float, default=0.25,
# #                        help='Confidence threshold for detections')
# #     parser.add_argument('--iou_threshold', type=float, default=0.45,
# #                        help='IoU threshold for NMS')
    
# #     args = parser.parse_args()
    
# #     # Initialize detector
# #     detector = ConveyorBeltDamageDetector(
# #         model_path=args.model_path,
# #         conf_threshold=args.conf_threshold,
# #         iou_threshold=args.iou_threshold
# #     )
    
# #     # Process directory
# #     detector.process_directory(args.image_dir, args.output_dir)
    
# #     print("\nPipeline execution completed!")


# # if __name__ == "__main__":
# #     main()


#!/usr/bin/env python3
"""
Fast Conveyor Belt Damage Detection Pipeline
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

class FastDetector:
    def __init__(self, min_area=200):
        self.min_area = min_area
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        height, width = image.shape[:2]
        
        # Resize for speed
        if width > 1024:
            scale = 1024 / width
            new_width = 1024
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            height, width = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = {}
        edge_threshold = 0.12
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area < self.min_area:
                continue
            
            is_near_edge = (x < width * edge_threshold or 
                           x + w > width * (1 - edge_threshold) or
                           y < height * edge_threshold or 
                           y + h > height * (1 - edge_threshold))
            
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            is_scratch = aspect_ratio > 3 and area > 100
            
            if is_near_edge:
                dtype = "edge_damage"
                conf = min(0.85, 0.6 + area / 3000)
            elif is_scratch:
                dtype = "scratch"
                conf = min(0.8, 0.55 + area / 2000)
            else:
                dtype = "surface_damage"
                conf = min(0.7, 0.5 + area / 2500)
            
            detections[str(len(detections) + 1)] = {
                "bbox_coordinates": [int(x), int(y), int(x + w), int(y + h)],
                "class": dtype,
                "confidence": float(conf)
            }
        
        # Draw on image
        annotated = image.copy()
        colors = {'scratch': (255, 165, 0), 'edge_damage': (0, 0, 255), 'surface_damage': (255, 0, 255)}
        
        for det in detections.values():
            x1, y1, x2, y2 = det['bbox_coordinates']
            color = colors.get(det['class'], (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, det['class'].replace('_', ' '), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated, detections

def main():
    parser = argparse.ArgumentParser(description='Conveyor Belt Damage Detection')
    parser.add_argument('--image_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_area', type=int, default=200, help='Minimum area to detect')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in Path(args.image_dir).iterdir() if f.suffix.lower() in exts]
    
    print(f"Found {len(images)} images")
    
    detector = FastDetector(min_area=args.min_area)
    
    for img_file in tqdm(images, desc="Processing"):
        try:
            annotated, detections = detector.detect(str(img_file))
            
            # Save
            cv2.imwrite(str(output_path / f"{img_file.stem}_annotated.jpg"), annotated)
            with open(output_path / f"{img_file.stem}.json", 'w') as f:
                json.dump(detections, f, indent=2)
        except Exception as e:
            print(f"Error on {img_file.name}: {e}")
    
    print(f"\n✅ Done! Output saved to {output_path}")

if __name__ == "__main__":
    main()