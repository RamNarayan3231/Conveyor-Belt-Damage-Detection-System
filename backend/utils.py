import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] coordinates
        box2: [x1, y1, x2, y2] coordinates
    
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_predictions_to_ground_truth(predictions: Dict, ground_truths: Dict, 
                                       iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth boxes using greedy assignment.
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    matched_gt = set()
    true_positives = []
    false_positives = []
    
    # Sort predictions by confidence (if available)
    pred_items = list(predictions.items())
    pred_items.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
    
    for pred_id, pred_data in pred_items:
        pred_box = pred_data['bbox_coordinates']
        best_iou = 0
        best_gt_id = None
        
        for gt_id, gt_data in ground_truths.items():
            if gt_id in matched_gt:
                continue
            gt_box = gt_data['bbox_coordinates']
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold and best_gt_id is not None:
            true_positives.append((pred_id, best_gt_id, best_iou))
            matched_gt.add(best_gt_id)
        else:
            false_positives.append(pred_id)
    
    false_negatives = [gt_id for gt_id in ground_truths.keys() if gt_id not in matched_gt]
    
    return true_positives, false_positives, false_negatives


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mf1(predictions: Dict, ground_truths: Dict) -> float:
    """
    Calculate mean F1 score across IoU thresholds from 0.5 to 0.95.
    
    Args:
        predictions: Dictionary of predictions
        ground_truths: Dictionary of ground truth annotations
    
    Returns:
        Mean F1 score
    """
    iou_thresholds = np.arange(0.50, 1.0, 0.05)
    f1_scores = []
    
    for iou_thresh in iou_thresholds:
        tp, fp, fn = match_predictions_to_ground_truth(predictions, ground_truths, iou_thresh)
        
        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
        
        f1 = calculate_f1_score(precision, recall)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def load_yolo_annotations(label_path: str, image_width: int, image_height: int) -> Dict:
    """
    Load YOLO format annotations and convert to pixel coordinates.
    
    Args:
        label_path: Path to YOLO format .txt file
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        Dictionary of bounding boxes in pixel coordinates
    """
    annotations = {}
    
    if not Path(label_path).exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) == 5:  # class_id, x_center, y_center, width, height
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Convert to pixel coordinates (x_min, y_min, x_max, y_max)
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)
                
                annotations[str(idx + 1)] = {
                    "bbox_coordinates": [x_min, y_min, x_max, y_max],
                    "class_id": int(class_id)
                }
    
    return annotations


def visualize_detections(image_path: str, detections: Dict, output_path: str = None):
    """Visualize detections on image."""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    colors = [(0, 0, 255), (0, 255, 255)]  # Red for class 0, Yellow for class 1
    
    for det_id, det_data in detections.items():
        x1, y1, x2, y2 = det_data['bbox_coordinates']
        class_id = det_data.get('class_id', 0)
        confidence = det_data.get('confidence', 0)
        
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"Damage: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow('Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()