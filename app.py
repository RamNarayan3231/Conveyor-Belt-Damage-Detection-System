"""
Conveyor Belt Damage Detection System
Optimized Fast Detection - Real-time performance
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import io
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="BeltGuard AI - Conveyor Belt Damage Detection",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .damage-box {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .scratch {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .edge-damage {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

class FastDamageDetector:
    """Fast damage detector optimized for speed"""
    
    def __init__(self):
        self.min_area = 200
        
    def fast_detect(self, image):
        """Fast detection using optimized algorithms"""
        height, width = image.shape[:2]
        
        # Resize for faster processing if image is large
        if width > 1024:
            scale = 1024 / width
            new_width = 1024
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            height, width = image.shape[:2]
        
        # Convert to grayscale (fast)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Fast edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = {}
        edge_threshold = 0.12  # 12% from edges
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter small detections
            if area < self.min_area:
                continue
            
            # Check if damage is near edges (for edge damage)
            is_near_edge = (x < width * edge_threshold or 
                           x + w > width * (1 - edge_threshold) or
                           y < height * edge_threshold or 
                           y + h > height * (1 - edge_threshold))
            
            # Check aspect ratio for scratches (long and thin)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            is_scratch_like = aspect_ratio > 3 and area > 100
            
            # Classify damage
            if is_near_edge:
                damage_type = "edge_damage"
                confidence = min(0.85, 0.6 + area / 3000)
            elif is_scratch_like:
                damage_type = "scratch"
                confidence = min(0.8, 0.55 + area / 2000)
            else:
                # General damage
                damage_type = "surface_damage"
                confidence = min(0.7, 0.5 + area / 2500)
            
            # Add detection
            det_id = len(detections) + 1
            detections[str(det_id)] = {
                "bbox_coordinates": [int(x), int(y), int(x + w), int(y + h)],
                "class": damage_type,
                "confidence": float(confidence)
            }
        
        # Draw on image
        annotated = image.copy()
        colors = {
            'scratch': (255, 165, 0),
            'edge_damage': (0, 0, 255),
            'surface_damage': (255, 0, 255)
        }
        
        for det in detections.values():
            x1, y1, x2, y2 = det['bbox_coordinates']
            color = colors.get(det['class'], (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{det['class'].replace('_', ' ')}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated, detections

def display_metrics(detections):
    """Display detection metrics"""
    if not detections:
        st.info("✅ No damage detected")
        return
    
    # Count by type
    damage_counts = {}
    for det in detections.values():
        dtype = det['class']
        damage_counts[dtype] = damage_counts.get(dtype, 0) + 1
    
    cols = st.columns(min(len(damage_counts) + 1, 4))
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total</h3>
            <h1 style="color: #667eea;">{len(detections)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    idx = 1
    colors = {'scratch': '#ff9800', 'edge_damage': '#f44336', 'surface_damage': '#9c27b0'}
    for dtype, count in damage_counts.items():
        if idx < len(cols):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{dtype.replace('_', ' ').title()}</h3>
                    <h1 style="color: {colors.get(dtype, '#667eea')};">{count}</h1>
                </div>
                """, unsafe_allow_html=True)
            idx += 1

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛠️ BeltGuard AI</h1>
        <p style="font-size: 1.2rem;">Fast Conveyor Belt Damage Detection</p>
        <p>Real-time detection of scratches and edge damage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=80)
        st.title("Settings")
        st.markdown("---")
        
        sensitivity = st.select_slider(
            "Detection Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        st.markdown("---")
        st.info("💡 **Quick Tips:**\n- Upload clear images\n- Supports JPG, PNG\n- Fast processing")
        
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    
    # Map sensitivity to min_area
    area_map = {"Low": 400, "Medium": 200, "High": 100}
    min_area = area_map[sensitivity]
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Process", "📊 History"])
    
    # Tab 1: Single Image
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="single"
        )
        
        if uploaded_file:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Original Image", width="stretch")
            
            # Detect button
            if st.button("🔍 Detect Damage", type="primary"):
                with st.spinner("Detecting..."):
                    start_time = time.time()
                    
                    # Detect
                    detector = FastDamageDetector()
                    detector.min_area = min_area
                    annotated, detections = detector.fast_detect(image)
                    
                    process_time = time.time() - start_time
                    
                    # Store in history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'filename': uploaded_file.name,
                        'detections': detections,
                        'time': process_time
                    })
                    
                    # Show results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                caption=f"Detected ({len(detections)} damages)", 
                                width="stretch")
                    
                    with col2:
                        st.success(f"✅ Processed in {process_time:.2f} seconds")
                        display_metrics(detections)
                        
                        if detections:
                            st.subheader("📍 Details")
                            for det_id, det in list(detections.items())[:5]:
                                x1, y1, x2, y2 = det['bbox_coordinates']
                                st.markdown(f"""
                                <div class="damage-box {det['class']}">
                                    <strong>{det_id}. {det['class'].replace('_', ' ').title()}</strong><br>
                                    Confidence: {det['confidence']:.0%}<br>
                                    Size: {x2-x1}×{y2-y1}px
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Download buttons
                            col_a, col_b = st.columns(2)
                            with col_a:
                                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                annotated_pil = Image.fromarray(annotated_rgb)
                                buf = io.BytesIO()
                                annotated_pil.save(buf, format="JPEG")
                                st.download_button("📥 Download Image", buf.getvalue(),
                                                 f"annotated_{uploaded_file.name}", "image/jpeg")
                            with col_b:
                                st.download_button("📄 Download JSON", 
                                                 json.dumps(detections, indent=2),
                                                 f"detections.json", "application/json")
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown('<div class="info-box">📁 Upload multiple images for batch processing</div>', 
                   unsafe_allow_html=True)
        
        files = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            key="batch"
        )
        
        if files and st.button("🚀 Process Batch", type="primary"):
            progress = st.progress(0)
            results = []
            detector = FastDamageDetector()
            detector.min_area = min_area
            
            for i, file in enumerate(files):
                progress.progress((i + 1) / len(files))
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                _, detections = detector.fast_detect(image)
                results.append({
                    'filename': file.name,
                    'damages': len(detections),
                    'types': list(set(d['class'] for d in detections.values()))
                })
            
            st.success("✅ Batch complete!")
            
            # Show summary
            df = pd.DataFrame(results)
            st.dataframe(df, width="stretch")
            
            # Download all results
            st.download_button("📦 Download All Results", 
                             json.dumps(results, indent=2),
                             "batch_results.json", "application/json")
    
    # Tab 3: History
    with tab3:
        if not st.session_state.detection_history:
            st.info("No history yet. Process some images first.")
        else:
            st.write(f"📊 Total: {len(st.session_state.detection_history)} analyses")
            
            for record in reversed(st.session_state.detection_history[-10:]):
                with st.expander(f"📸 {record['timestamp'].strftime('%H:%M:%S')} - {record['filename']}"):
                    st.write(f"Damages found: {len(record['detections'])}")
                    st.write(f"Processing time: {record['time']:.2f}s")
                    if record['detections']:
                        st.json(record['detections'])

if __name__ == "__main__":
    main()


# """
# Conveyor Belt Damage Detection System
# Advanced Computer Vision-based detection - No training required!
# """

# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import json
# import io
# from datetime import datetime
# import plotly.graph_objects as go
# from PIL import Image
# import sys
# import os

# # Page configuration
# st.set_page_config(
#     page_title="BeltGuard AI - Conveyor Belt Damage Detection",
#     page_icon="🛠️",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 1rem;
#         margin-bottom: 2rem;
#         text-align: center;
#         color: white;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 1rem;
#         border-radius: 1rem;
#         text-align: center;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .damage-box {
#         padding: 0.5rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .scratch {
#         background-color: #fff3e0;
#         border-left: 4px solid #ff9800;
#     }
#     .edge-damage {
#         background-color: #ffebee;
#         border-left: 4px solid #f44336;
#     }
#     .info-box {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #2196f3;
#         margin: 1rem 0;
#     }
#     .damage-highlight {
#         background-color: #ffeb3b;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.3rem;
#         font-weight: bold;
#     }
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.5rem 2rem;
#         border-radius: 0.5rem;
#         font-weight: bold;
#         width: 100%;
#     }
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
#         transition: all 0.3s;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'detection_history' not in st.session_state:
#     st.session_state.detection_history = []

# class AdvancedDamageDetector:
#     """Advanced damage detector using computer vision techniques"""
    
#     def __init__(self):
#         self.min_line_length = 50
#         self.min_area = 300
        
#     def enhance_contrast(self, image):
#         """Enhance image contrast for better detection"""
#         # Convert to LAB color space
#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
        
#         # Apply CLAHE to L channel
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         l_enhanced = clahe.apply(l)
        
#         # Merge back
#         enhanced_lab = cv2.merge([l_enhanced, a, b])
#         enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
#         return enhanced
    
#     def detect_scratches(self, image):
#         """Detect scratches using line detection"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Use adaptive threshold
#         thresh = cv2.adaptiveThreshold(blurred, 255, 
#                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Use Hough Line Transform to detect lines
#         lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, 
#                                 minLineLength=self.min_line_length, maxLineGap=10)
        
#         scratches = []
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
#                 # Filter for significant scratches
#                 if length > self.min_line_length:
#                     # Create bounding box with padding
#                     padding = 15
#                     x_min = max(0, min(x1, x2) - padding)
#                     y_min = max(0, min(y1, y2) - padding)
#                     x_max = min(image.shape[1], max(x1, x2) + padding)
#                     y_max = min(image.shape[0], max(y1, y2) + padding)
                    
#                     # Calculate confidence based on length
#                     confidence = min(0.95, 0.6 + (length / 500))
                    
#                     scratches.append({
#                         "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
#                         "confidence": confidence,
#                         "length": length,
#                         "type": "scratch"
#                     })
        
#         # Remove overlapping detections
#         scratches = self.non_max_suppression(scratches, 0.4)
        
#         return scratches
    
#     def detect_edge_damage(self, image):
#         """Detect damage on belt edges"""
#         height, width = image.shape[:2]
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Edge detection
#         edges = cv2.Canny(gray, 50, 150)
        
#         # Find contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         edge_damages = []
#         edge_threshold = 0.15  # 15% from edges
        
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             area = w * h
            
#             # Check if damage is near edges
#             is_left_edge = x < width * edge_threshold
#             is_right_edge = x + w > width * (1 - edge_threshold)
#             is_top_edge = y < height * edge_threshold
#             is_bottom_edge = y + h > height * (1 - edge_threshold)
            
#             is_edge_damage = is_left_edge or is_right_edge or is_top_edge or is_bottom_edge
            
#             # Check if area is significant and it's on edge
#             if is_edge_damage and area > self.min_area:
#                 # Calculate confidence
#                 confidence = min(0.9, 0.5 + (area / 2000))
                
#                 edge_damages.append({
#                     "bbox": [int(x), int(y), int(x + w), int(y + h)],
#                     "confidence": confidence,
#                     "area": area,
#                     "type": "edge_damage",
#                     "edge": "left" if is_left_edge else "right" if is_right_edge else "top" if is_bottom_edge else "bottom"
#                 })
        
#         # Remove overlapping
#         edge_damages = self.non_max_suppression(edge_damages, 0.5)
        
#         return edge_damages
    
#     def detect_texture_anomalies(self, image):
#         """Detect texture anomalies that might indicate damage"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply Local Binary Pattern (LBP) like effect using variance
#         kernel_size = 15
#         kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
#         # Local mean and variance
#         local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
#         local_sq_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
#         local_variance = local_sq_mean - (local_mean ** 2)
        
#         # Normalize variance
#         local_variance = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX)
#         local_variance = local_variance.astype(np.uint8)
        
#         # Threshold to find anomalies
#         _, anomaly_mask = cv2.threshold(local_variance, 200, 255, cv2.THRESH_BINARY)
        
#         # Find contours of anomalies
#         contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         anomalies = []
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             area = w * h
            
#             if self.min_area < area < 5000:  # Filter by size
#                 anomalies.append({
#                     "bbox": [int(x), int(y), int(x + w), int(y + h)],
#                     "confidence": 0.7,
#                     "area": area,
#                     "type": "texture_anomaly"
#                 })
        
#         return anomalies
    
#     def non_max_suppression(self, detections, iou_threshold):
#         """Apply Non-Maximum Suppression"""
#         if not detections:
#             return []
        
#         # Sort by confidence
#         detections.sort(key=lambda x: x['confidence'], reverse=True)
        
#         keep = []
#         while detections:
#             best = detections.pop(0)
#             keep.append(best)
            
#             to_remove = []
#             for i, det in enumerate(detections):
#                 iou = self.calculate_iou(best['bbox'], det['bbox'])
#                 if iou > iou_threshold:
#                     to_remove.append(i)
            
#             # Remove overlapping detections
#             for idx in reversed(to_remove):
#                 detections.pop(idx)
        
#         return keep
    
#     def calculate_iou(self, box1, box2):
#         """Calculate Intersection over Union"""
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         intersection = max(0, x2 - x1) * max(0, y2 - y1)
#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = area1 + area2 - intersection
        
#         return intersection / union if union > 0 else 0
    
#     def detect_damage(self, image):
#         """
#         Main detection function that combines all methods
#         """
#         # Enhance image first
#         enhanced = self.enhance_contrast(image)
        
#         # Run all detection methods
#         scratches = self.detect_scratches(enhanced)
#         edge_damages = self.detect_edge_damage(enhanced)
#         anomalies = self.detect_texture_anomalies(enhanced)
        
#         # Combine all detections
#         all_detections = scratches + edge_damages + anomalies
        
#         # Apply final NMS across all detections
#         final_detections = self.non_max_suppression(all_detections, 0.4)
        
#         # Convert to required format
#         detections_dict = {}
#         for i, det in enumerate(final_detections):
#             detections_dict[str(i + 1)] = {
#                 "bbox_coordinates": det['bbox'],
#                 "class": det['type'].replace('_', ' '),
#                 "confidence": det['confidence']
#             }
        
#         # Draw detections on image
#         annotated = image.copy()
#         colors = {
#             'scratch': (255, 165, 0),      # Orange
#             'edge_damage': (0, 0, 255),    # Red
#             'texture_anomaly': (255, 0, 255)  # Purple
#         }
        
#         for det in final_detections:
#             x1, y1, x2, y2 = det['bbox']
#             color = colors.get(det['type'], (0, 255, 0))
            
#             # Draw rectangle
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
#             # Add label
#             label = f"{det['type'].upper()}: {det['confidence']:.0%}"
#             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#             cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
#                          (x1 + label_size[0], y1), color, -1)
#             cv2.putText(annotated, label, (x1, y1 - 5),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         return annotated, detections_dict

# def display_metrics(detections):
#     """Display detection metrics"""
#     if not detections:
#         st.info("✅ No damage detected in this image")
#         return
    
#     # Count by type
#     damage_counts = {}
#     for det in detections.values():
#         damage_type = det['class']
#         damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    
#     cols = st.columns(min(len(damage_counts) + 1, 4))
    
#     # Total damages
#     with cols[0]:
#         st.markdown(f"""
#         <div class="metric-card">
#             <h3>Total Damages</h3>
#             <h1 style="color: #667eea;">{len(detections)}</h1>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Individual damage types
#     col_idx = 1
#     for damage_type, count in damage_counts.items():
#         if col_idx < len(cols):
#             color = "#ff9800" if "scratch" in damage_type else "#f44336" if "edge" in damage_type else "#9c27b0"
#             with cols[col_idx]:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h3>{damage_type.title()}</h3>
#                     <h1 style="color: {color};">{count}</h1>
#                 </div>
#                 """, unsafe_allow_html=True)
#             col_idx += 1

# def create_damage_heatmap(image, detections):
#     """Create a heatmap of damage locations"""
#     if not detections:
#         return None
    
#     heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
#     for det in detections.values():
#         x1, y1, x2, y2 = det['bbox_coordinates']
#         confidence = det['confidence']
#         heatmap[y1:y2, x1:x2] += confidence
    
#     # Normalize heatmap
#     if heatmap.max() > 0:
#         heatmap = heatmap / heatmap.max()
    
#     # Apply colormap
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
#     # Overlay on image
#     overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
#     return overlay

# def main():
#     # Header
#     st.markdown("""
#     <div class="main-header">
#         <h1>🛠️ BeltGuard AI</h1>
#         <p style="font-size: 1.2rem;">Advanced Conveyor Belt Damage Detection - No Training Required!</p>
#         <p>Using State-of-the-Art Computer Vision Algorithms</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=80)
#         st.title("BeltGuard AI")
#         st.markdown("---")
        
#         st.subheader("📊 Detection Methods")
#         st.markdown("""
#         **Advanced CV Techniques:**
#         - ✅ Edge Detection (Canny)
#         - ✅ Line Detection (Hough Transform)
#         - ✅ Texture Analysis
#         - ✅ Contrast Enhancement (CLAHE)
#         - ✅ Morphological Operations
#         """)
        
#         st.markdown("---")
        
#         st.subheader("⚙️ Settings")
#         sensitivity = st.select_slider(
#             "Detection Sensitivity",
#             options=["Low", "Medium", "High", "Very High"],
#             value="Medium"
#         )
        
#         st.markdown("---")
#         st.info("💡 **Tip:** Upload clear images of conveyor belts for best results.")
    
#     # Main content
#     tab1, tab2, tab3 = st.tabs([
#         "📸 Single Image Detection",
#         "📁 Batch Processing",
#         "📊 Detection Analytics"
#     ])
    
#     # Tab 1: Single Image Detection
#     with tab1:
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.subheader("Upload Image")
#             uploaded_file = st.file_uploader(
#                 "Choose a conveyor belt image",
#                 type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
#                 key="single_upload"
#             )
            
#             if uploaded_file is not None:
#                 # Read and display image
#                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#                 image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
#                 # Convert BGR to RGB for display
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 st.image(image_rgb, caption="Original Image", use_container_width=True)
                
#                 # Detect button
#                 if st.button("🔍 Detect Damage", key="detect_single", type="primary"):
#                     with st.spinner("🔬 Analyzing image with advanced CV algorithms..."):
#                         # Initialize detector
#                         detector = AdvancedDamageDetector()
                        
#                         # Update sensitivity parameters
#                         if sensitivity == "Low":
#                             detector.min_line_length = 80
#                             detector.min_area = 500
#                         elif sensitivity == "Medium":
#                             detector.min_line_length = 50
#                             detector.min_area = 300
#                         elif sensitivity == "High":
#                             detector.min_line_length = 30
#                             detector.min_area = 200
#                         else:  # Very High
#                             detector.min_line_length = 20
#                             detector.min_area = 100
                        
#                         # Run detection
#                         annotated_image, detections = detector.detect_damage(image)
                        
#                         # Store in history
#                         st.session_state.detection_history.append({
#                             'timestamp': datetime.now(),
#                             'filename': uploaded_file.name,
#                             'detections': detections,
#                             'image': annotated_image,
#                             'original': image
#                         })
                        
#                         # Display results
#                         with col2:
#                             st.subheader("🔍 Detection Results")
#                             annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#                             st.image(annotated_rgb, caption="Detected Damages", use_container_width=True)
                            
#                             # Metrics
#                             display_metrics(detections)
                            
#                             # Detailed detections
#                             if detections:
#                                 st.subheader("📍 Damage Locations")
#                                 for det_id, det in detections.items():
#                                     x1, y1, x2, y2 = det['bbox_coordinates']
#                                     damage_class = det['class'].title()
#                                     confidence = det['confidence']
                                    
#                                     st.markdown(f"""
#                                     <div class="damage-box {det['class'].replace(' ', '-')}">
#                                         <strong>🔴 Damage {det_id}</strong><br>
#                                         Type: <span class="damage-highlight">{damage_class}</span><br>
#                                         Confidence: {confidence:.1%}<br>
#                                         Location: [{x1}, {y1}, {x2}, {y2}]<br>
#                                         Size: {x2-x1} × {y2-y1} px
#                                     </div>
#                                     """, unsafe_allow_html=True)
                                
#                                 # Export options
#                                 col_export1, col_export2 = st.columns(2)
#                                 with col_export1:
#                                     annotated_pil = Image.fromarray(annotated_rgb)
#                                     buf = io.BytesIO()
#                                     annotated_pil.save(buf, format="JPEG", quality=95)
#                                     st.download_button(
#                                         label="📥 Download Annotated Image",
#                                         data=buf.getvalue(),
#                                         file_name=f"annotated_{uploaded_file.name}",
#                                         mime="image/jpeg"
#                                     )
                                
#                                 with col_export2:
#                                     json_str = json.dumps(detections, indent=2)
#                                     st.download_button(
#                                         label="📄 Download JSON",
#                                         data=json_str,
#                                         file_name=f"detections_{Path(uploaded_file.name).stem}.json",
#                                         mime="application/json"
#                                     )
                                
#                                 # Show damage heatmap
#                                 st.subheader("🔥 Damage Heatmap")
#                                 heatmap = create_damage_heatmap(image, detections)
#                                 if heatmap is not None:
#                                     heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#                                     st.image(heatmap_rgb, caption="Damage Intensity Map", use_container_width=True)
#                             else:
#                                 st.success("✅ No damage detected in this image!")
    
#     # Tab 2: Batch Processing
#     with tab2:
#         st.subheader("Batch Processing")
#         st.markdown('<div class="info-box">📁 Upload multiple images for batch processing. Results will be saved together.</div>', unsafe_allow_html=True)
        
#         uploaded_files = st.file_uploader(
#             "Choose multiple images",
#             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
#             accept_multiple_files=True,
#             key="batch_upload"
#         )
        
#         if uploaded_files:
#             st.write(f"📁 {len(uploaded_files)} files selected")
            
#             if st.button("🚀 Process All Images", key="process_batch", type="primary"):
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
#                 batch_results = []
                
#                 detector = AdvancedDamageDetector()
                
#                 # Update sensitivity
#                 if sensitivity == "Low":
#                     detector.min_line_length = 80
#                     detector.min_area = 500
#                 elif sensitivity == "Medium":
#                     detector.min_line_length = 50
#                     detector.min_area = 300
#                 elif sensitivity == "High":
#                     detector.min_line_length = 30
#                     detector.min_area = 200
#                 else:
#                     detector.min_line_length = 20
#                     detector.min_area = 100
                
#                 for idx, file in enumerate(uploaded_files):
#                     status_text.text(f"Processing {file.name}...")
                    
#                     # Process image
#                     file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
#                     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#                     annotated_image, detections = detector.detect_damage(image)
                    
#                     batch_results.append({
#                         'filename': file.name,
#                         'detections': detections,
#                         'image': annotated_image
#                     })
                    
#                     progress_bar.progress((idx + 1) / len(uploaded_files))
                
#                 status_text.text("✅ Batch processing completed!")
                
#                 # Summary table
#                 st.subheader("Batch Results Summary")
#                 summary_data = []
#                 for result in batch_results:
#                     scratches = sum(1 for d in result['detections'].values() if 'scratch' in d['class'])
#                     edge_damages = sum(1 for d in result['detections'].values() if 'edge' in d['class'])
#                     summary_data.append({
#                         'Filename': result['filename'],
#                         'Total Damages': len(result['detections']),
#                         'Scratches': scratches,
#                         'Edge Damage': edge_damages
#                     })
                
#                 summary_df = pd.DataFrame(summary_data)
#                 st.dataframe(summary_df, use_container_width=True)
                
#                 # Download all results
#                 all_results_json = json.dumps([
#                     {'filename': r['filename'], 'detections': r['detections']} 
#                     for r in batch_results
#                 ], indent=2)
                
#                 st.download_button(
#                     label="📦 Download All Results (JSON)",
#                     data=all_results_json,
#                     file_name="all_detections.json",
#                     mime="application/json"
#                 )
    
#     # Tab 3: Detection Analytics
#     with tab3:
#         st.subheader("Detection Analytics")
        
#         if not st.session_state.detection_history:
#             st.info("No detection history yet. Process some images to see analytics.")
#         else:
#             # Statistics
#             total_images = len(st.session_state.detection_history)
#             total_damages = sum(len(r['detections']) for r in st.session_state.detection_history)
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Total Images Analyzed", total_images)
#             with col2:
#                 st.metric("Total Damages Found", total_damages)
#             with col3:
#                 avg_damage = total_damages / total_images if total_images > 0 else 0
#                 st.metric("Average Damages per Image", f"{avg_damage:.1f}")
            
#             # Damage type distribution
#             damage_types = {}
#             for record in st.session_state.detection_history:
#                 for det in record['detections'].values():
#                     damage_type = det['class']
#                     damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
            
#             if damage_types:
#                 fig = go.Figure(data=[go.Pie(
#                     labels=list(damage_types.keys()),
#                     values=list(damage_types.values()),
#                     hole=0.4,
#                     marker_colors=['#ff9800', '#f44336', '#9c27b0']
#                 )])
#                 fig.update_layout(title="Damage Type Distribution", height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             # Recent detections
#             st.subheader("Recent Detections")
#             for record in reversed(st.session_state.detection_history[-5:]):
#                 with st.expander(f"📸 {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {record['filename']}"):
#                     col1, col2 = st.columns([1, 1])
#                     with col1:
#                         annotated_rgb = cv2.cvtColor(record['image'], cv2.COLOR_BGR2RGB)
#                         st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
#                     with col2:
#                         display_metrics(record['detections'])

# if __name__ == "__main__":
#     main()


# # """
# # Conveyor Belt Damage Detection System
# # Advanced Computer Vision-based detection - No training required!
# # """

# # import streamlit as st
# # import cv2
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # import json
# # import io
# # from datetime import datetime
# # import plotly.graph_objects as go
# # from PIL import Image
# # import sys
# # import os

# # # Page configuration
# # st.set_page_config(
# #     page_title="BeltGuard AI - Conveyor Belt Damage Detection",
# #     page_icon="🛠️",
# #     layout="wide"
# # )

# # # Custom CSS
# # st.markdown("""
# # <style>
# #     .main-header {
# #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #         padding: 2rem;
# #         border-radius: 1rem;
# #         margin-bottom: 2rem;
# #         text-align: center;
# #         color: white;
# #     }
# #     .metric-card {
# #         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
# #         padding: 1rem;
# #         border-radius: 1rem;
# #         text-align: center;
# #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# #     }
# #     .damage-box {
# #         padding: 0.5rem;
# #         border-radius: 0.5rem;
# #         margin: 0.5rem 0;
# #     }
# #     .scratch {
# #         background-color: #fff3e0;
# #         border-left: 4px solid #ff9800;
# #     }
# #     .edge-damage {
# #         background-color: #ffebee;
# #         border-left: 4px solid #f44336;
# #     }
# #     .info-box {
# #         background-color: #e3f2fd;
# #         padding: 1rem;
# #         border-radius: 0.5rem;
# #         border-left: 4px solid #2196f3;
# #         margin: 1rem 0;
# #     }
# #     .damage-highlight {
# #         background-color: #ffeb3b;
# #         padding: 0.2rem 0.5rem;
# #         border-radius: 0.3rem;
# #         font-weight: bold;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # Initialize session state
# # if 'detection_history' not in st.session_state:
# #     st.session_state.detection_history = []

# # class AdvancedDamageDetector:
# #     """Advanced damage detector using computer vision techniques"""
    
# #     def __init__(self):
# #         self.kernel_size = 5
# #         self.sobel_kernel = 3
        
# #     def enhance_contrast(self, image):
# #         """Enhance image contrast for better detection"""
# #         # Convert to LAB color space
# #         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# #         l, a, b = cv2.split(lab)
        
# #         # Apply CLAHE to L channel
# #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #         l_enhanced = clahe.apply(l)
        
# #         # Merge back
# #         enhanced_lab = cv2.merge([l_enhanced, a, b])
# #         enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
# #         return enhanced
    
# #     def detect_edges(self, image):
# #         """Advanced edge detection for finding damage"""
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# #         # Apply multiple edge detection techniques
# #         # 1. Canny edge detection
# #         edges_canny = cv2.Canny(gray, 30, 100)
        
# #         # 2. Sobel edge detection
# #         sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# #         sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# #         edges_sobel = cv2.magnitude(sobelx, sobely)
# #         edges_sobel = np.uint8(np.clip(edges_sobel, 0, 255))
        
# #         # 3. Laplacian edge detection
# #         edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# #         edges_laplacian = np.uint8(np.abs(edges_laplacian))
        
# #         # Combine all edge detections
# #         edges_combined = cv2.bitwise_or(edges_canny, edges_sobel)
# #         edges_combined = cv2.bitwise_or(edges_combined, edges_laplacian)
        
# #         # Apply morphological operations to connect edges
# #         kernel = np.ones((3,3), np.uint8)
# #         edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
        
# #         return edges_combined
    
# #     def detect_scratches(self, image):
# #         """Detect scratches using line detection"""
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# #         # Apply Gaussian blur to reduce noise
# #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
# #         # Use adaptive threshold
# #         thresh = cv2.adaptiveThreshold(blurred, 255, 
# #                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                        cv2.THRESH_BINARY_INV, 11, 2)
        
# #         # Use Hough Line Transform to detect lines
# #         lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, 
# #                                 minLineLength=30, maxLineGap=10)
        
# #         scratches = []
# #         if lines is not None:
# #             for line in lines:
# #                 x1, y1, x2, y2 = line[0]
# #                 length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
# #                 # Filter for significant scratches
# #                 if length > 40:
# #                     # Create bounding box with padding
# #                     padding = 15
# #                     x_min = max(0, min(x1, x2) - padding)
# #                     y_min = max(0, min(y1, y2) - padding)
# #                     x_max = min(image.shape[1], max(x1, x2) + padding)
# #                     y_max = min(image.shape[0], max(y1, y2) + padding)
                    
# #                     # Calculate confidence based on length and contrast
# #                     confidence = min(0.95, 0.6 + (length / 500))
                    
# #                     scratches.append({
# #                         "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
# #                         "confidence": confidence,
# #                         "length": length,
# #                         "type": "scratch"
# #                     })
        
# #         # Remove overlapping detections
# #         scratches = self.non_max_suppression(scratches, 0.4)
        
# #         return scratches
    
# #     def detect_edge_damage(self, image):
# #         """Detect damage on belt edges"""
# #         height, width = image.shape[:2]
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# #         # Edge detection
# #         edges = cv2.Canny(gray, 50, 150)
        
# #         # Find contours
# #         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# #         edge_damages = []
# #         edge_threshold = 0.15  # 15% from edges
        
# #         for contour in contours:
# #             x, y, w, h = cv2.boundingRect(contour)
# #             area = w * h
            
# #             # Check if damage is near edges
# #             is_left_edge = x < width * edge_threshold
# #             is_right_edge = x + w > width * (1 - edge_threshold)
# #             is_top_edge = y < height * edge_threshold
# #             is_bottom_edge = y + h > height * (1 - edge_threshold)
            
# #             is_edge_damage = is_left_edge or is_right_edge or is_top_edge or is_bottom_edge
            
# #             # Check if area is significant and it's on edge
# #             if is_edge_damage and area > 300:
# #                 # Calculate confidence
# #                 confidence = min(0.9, 0.5 + (area / 2000))
                
# #                 edge_damages.append({
# #                     "bbox": [int(x), int(y), int(x + w), int(y + h)],
# #                     "confidence": confidence,
# #                     "area": area,
# #                     "type": "edge_damage",
# #                     "edge": "left" if is_left_edge else "right" if is_right_edge else "top" if is_top_edge else "bottom"
# #                 })
        
# #         # Remove overlapping
# #         edge_damages = self.non_max_suppression(edge_damages, 0.5)
        
# #         return edge_damages
    
# #     def detect_texture_anomalies(self, image):
# #         """Detect texture anomalies that might indicate damage"""
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# #         # Apply Local Binary Pattern (LBP) like effect using variance
# #         kernel_size = 15
# #         kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
# #         # Local mean and variance
# #         local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
# #         local_sq_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
# #         local_variance = local_sq_mean - (local_mean ** 2)
        
# #         # Normalize variance
# #         local_variance = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX)
# #         local_variance = local_variance.astype(np.uint8)
        
# #         # Threshold to find anomalies
# #         _, anomaly_mask = cv2.threshold(local_variance, 200, 255, cv2.THRESH_BINARY)
        
# #         # Find contours of anomalies
# #         contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# #         anomalies = []
# #         for contour in contours:
# #             x, y, w, h = cv2.boundingRect(contour)
# #             area = w * h
            
# #             if 200 < area < 5000:  # Filter by size
# #                 anomalies.append({
# #                     "bbox": [int(x), int(y), int(x + w), int(y + h)],
# #                     "confidence": 0.7,
# #                     "area": area,
# #                     "type": "texture_anomaly"
# #                 })
        
# #         return anomalies
    
# #     def non_max_suppression(self, detections, iou_threshold):
# #         """Apply Non-Maximum Suppression"""
# #         if not detections:
# #             return []
        
# #         # Sort by confidence
# #         detections.sort(key=lambda x: x['confidence'], reverse=True)
        
# #         keep = []
# #         while detections:
# #             best = detections.pop(0)
# #             keep.append(best)
            
# #             to_remove = []
# #             for i, det in enumerate(detections):
# #                 iou = self.calculate_iou(best['bbox'], det['bbox'])
# #                 if iou > iou_threshold:
# #                     to_remove.append(i)
            
# #             # Remove overlapping detections
# #             for idx in reversed(to_remove):
# #                 detections.pop(idx)
        
# #         return keep
    
# #     def calculate_iou(self, box1, box2):
# #         """Calculate Intersection over Union"""
# #         x1 = max(box1[0], box2[0])
# #         y1 = max(box1[1], box2[1])
# #         x2 = min(box1[2], box2[2])
# #         y2 = min(box1[3], box2[3])
        
# #         intersection = max(0, x2 - x1) * max(0, y2 - y1)
# #         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
# #         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
# #         union = area1 + area2 - intersection
        
# #         return intersection / union if union > 0 else 0
    
# #     def detect_damage(self, image):
# #         """
# #         Main detection function that combines all methods
# #         """
# #         # Enhance image first
# #         enhanced = self.enhance_contrast(image)
        
# #         # Run all detection methods
# #         scratches = self.detect_scratches(enhanced)
# #         edge_damages = self.detect_edge_damage(enhanced)
# #         anomalies = self.detect_texture_anomalies(enhanced)
        
# #         # Combine all detections
# #         all_detections = scratches + edge_damages + anomalies
        
# #         # Apply final NMS across all detections
# #         final_detections = self.non_max_suppression(all_detections, 0.4)
        
# #         # Convert to required format
# #         detections_dict = {}
# #         for i, det in enumerate(final_detections):
# #             detections_dict[str(i + 1)] = {
# #                 "bbox_coordinates": det['bbox'],
# #                 "class": det['type'].replace('_', ' '),
# #                 "confidence": det['confidence']
# #             }
        
# #         # Draw detections on image
# #         annotated = image.copy()
# #         colors = {
# #             'scratch': (255, 165, 0),      # Orange
# #             'edge_damage': (0, 0, 255),    # Red
# #             'texture_anomaly': (255, 0, 255)  # Purple
# #         }
        
# #         for det in final_detections:
# #             x1, y1, x2, y2 = det['bbox']
# #             color = colors.get(det['type'], (0, 255, 0))
            
# #             # Draw rectangle
# #             cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
# #             # Add label
# #             label = f"{det['type'].upper()}: {det['confidence']:.0%}"
# #             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
# #             cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
# #                          (x1 + label_size[0], y1), color, -1)
# #             cv2.putText(annotated, label, (x1, y1 - 5),
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
# #         return annotated, detections_dict

# # def display_metrics(detections):
# #     """Display detection metrics"""
# #     if not detections:
# #         st.info("No damage detected in this image")
# #         return
    
# #     # Count by type
# #     damage_counts = {}
# #     for det in detections.values():
# #         damage_type = det['class']
# #         damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    
# #     cols = st.columns(len(damage_counts) + 1)
    
# #     # Total damages
# #     with cols[0]:
# #         st.markdown(f"""
# #         <div class="metric-card">
# #             <h3>Total Damages</h3>
# #             <h1 style="color: #667eea;">{len(detections)}</h1>
# #         </div>
# #         """, unsafe_allow_html=True)
    
# #     # Individual damage types
# #     for idx, (damage_type, count) in enumerate(damage_counts.items(), 1):
# #         color = "#ff9800" if "scratch" in damage_type else "#f44336" if "edge" in damage_type else "#9c27b0"
# #         with cols[idx]:
# #             st.markdown(f"""
# #             <div class="metric-card">
# #                 <h3>{damage_type.title()}</h3>
# #                 <h1 style="color: {color};">{count}</h1>
# #             </div>
# #             """, unsafe_allow_html=True)

# # def create_damage_heatmap(image, detections):
# #     """Create a heatmap of damage locations"""
# #     if not detections:
# #         return None
    
# #     heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
# #     for det in detections.values():
# #         x1, y1, x2, y2 = det['bbox_coordinates']
# #         confidence = det['confidence']
# #         heatmap[y1:y2, x1:x2] += confidence
    
# #     # Normalize heatmap
# #     if heatmap.max() > 0:
# #         heatmap = heatmap / heatmap.max()
    
# #     # Apply colormap
# #     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
# #     # Overlay on image
# #     overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
# #     return overlay

# # def main():
# #     # Header
# #     st.markdown("""
# #     <div class="main-header">
# #         <h1>🛠️ BeltGuard AI</h1>
# #         <p style="font-size: 1.2rem;">Advanced Conveyor Belt Damage Detection - No Training Required!</p>
# #         <p>Using State-of-the-Art Computer Vision Algorithms</p>
# #     </div>
# #     """, unsafe_allow_html=True)
    
# #     # Sidebar
# #     with st.sidebar:
# #         st.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=80)
# #         st.title("BeltGuard AI")
# #         st.markdown("---")
        
# #         st.subheader("📊 Detection Methods")
# #         st.markdown("""
# #         **Advanced CV Techniques:**
# #         - ✅ Edge Detection (Canny/Sobel)
# #         - ✅ Line Detection (Hough Transform)
# #         - ✅ Texture Analysis
# #         - ✅ Contrast Enhancement (CLAHE)
# #         - ✅ Morphological Operations
# #         """)
        
# #         st.markdown("---")
        
# #         st.subheader("⚙️ Settings")
# #         sensitivity = st.select_slider(
# #             "Detection Sensitivity",
# #             options=["Low", "Medium", "High", "Very High"],
# #             value="Medium"
# #         )
        
# #         sensitivity_map = {
# #             "Low": {"min_length": 80, "min_area": 500},
# #             "Medium": {"min_length": 50, "min_area": 300},
# #             "High": {"min_length": 30, "min_area": 200},
# #             "Very High": {"min_length": 20, "min_area": 100}
# #         }
        
# #         st.markdown("---")
# #         st.info("💡 **Tip:** Upload clear images of conveyor belts for best results. The system detects scratches (surface lines) and edge damage (belt boundary issues).")
    
# #     # Main content
# #     tab1, tab2, tab3 = st.tabs([
# #         "📸 Single Image Detection",
# #         "📁 Batch Processing",
# #         "📊 Detection Analytics"
# #     ])
    
# #     # Tab 1: Single Image Detection
# #     with tab1:
# #         col1, col2 = st.columns([1, 1])
        
# #         with col1:
# #             st.subheader("Upload Image")
# #             uploaded_file = st.file_uploader(
# #                 "Choose a conveyor belt image",
# #                 type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
# #                 key="single_upload"
# #             )
            
# #             if uploaded_file is not None:
# #                 # Read and display image
# #                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
# #                 image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# #                 st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
# #                         caption="Original Image", width=None)
                
# #                 # Detect button
# #                 if st.button("🔍 Detect Damage", key="detect_single", type="primary"):
# #                     with st.spinner("🔬 Analyzing image with advanced CV algorithms..."):
# #                         # Initialize detector
# #                         detector = AdvancedDamageDetector()
                        
# #                         # Update sensitivity parameters
# #                         detector.min_line_length = sensitivity_map[sensitivity]["min_length"]
# #                         detector.min_area = sensitivity_map[sensitivity]["min_area"]
                        
# #                         # Run detection
# #                         annotated_image, detections = detector.detect_damage(image)
                        
# #                         # Store in history
# #                         st.session_state.detection_history.append({
# #                             'timestamp': datetime.now(),
# #                             'filename': uploaded_file.name,
# #                             'detections': detections,
# #                             'image': annotated_image,
# #                             'original': image
# #                         })
                        
# #                         # Display results
# #                         with col2:
# #                             st.subheader("🔍 Detection Results")
# #                             st.image(annotated_image, caption="Detected Damages", width=None)
                            
# #                             # Metrics
# #                             display_metrics(detections)
                            
# #                             # Detailed detections
# #                             if detections:
# #                                 st.subheader("📍 Damage Locations")
# #                                 for det_id, det in detections.items():
# #                                     x1, y1, x2, y2 = det['bbox_coordinates']
# #                                     damage_class = det['class'].title()
# #                                     confidence = det['confidence']
                                    
# #                                     st.markdown(f"""
# #                                     <div class="damage-box {det['class'].replace(' ', '-')}">
# #                                         <strong>🔴 Damage {det_id}</strong><br>
# #                                         Type: <span class="damage-highlight">{damage_class}</span><br>
# #                                         Confidence: {confidence:.1%}<br>
# #                                         Location: [{x1}, {y1}, {x2}, {y2}]<br>
# #                                         Size: {x2-x1} × {y2-y1} px
# #                                     </div>
# #                                     """, unsafe_allow_html=True)
                                
# #                                 # Export options
# #                                 col_export1, col_export2 = st.columns(2)
# #                                 with col_export1:
# #                                     annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
# #                                     buf = io.BytesIO()
# #                                     annotated_pil.save(buf, format="JPEG", quality=95)
# #                                     st.download_button(
# #                                         label="📥 Download Annotated Image",
# #                                         data=buf.getvalue(),
# #                                         file_name=f"annotated_{uploaded_file.name}",
# #                                         mime="image/jpeg"
# #                                     )
                                
# #                                 with col_export2:
# #                                     json_str = json.dumps(detections, indent=2)
# #                                     st.download_button(
# #                                         label="📄 Download JSON",
# #                                         data=json_str,
# #                                         file_name=f"detections_{Path(uploaded_file.name).stem}.json",
# #                                         mime="application/json"
# #                                     )
                                
# #                                 # Show damage heatmap
# #                                 st.subheader("🔥 Damage Heatmap")
# #                                 heatmap = create_damage_heatmap(image, detections)
# #                                 if heatmap is not None:
# #                                     st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 
# #                                             caption="Damage Intensity Map", width=None)
# #                             else:
# #                                 st.success("✅ No damage detected in this image!")
    
# #     # Tab 2: Batch Processing
# #     with tab2:
# #         st.subheader("Batch Processing")
# #         st.markdown('<div class="info-box">📁 Upload multiple images for batch processing. Results will be saved together.</div>', unsafe_allow_html=True)
        
# #         uploaded_files = st.file_uploader(
# #             "Choose multiple images",
# #             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
# #             accept_multiple_files=True,
# #             key="batch_upload"
# #         )
        
# #         if uploaded_files:
# #             st.write(f"📁 {len(uploaded_files)} files selected")
            
# #             if st.button("🚀 Process All Images", key="process_batch", type="primary"):
# #                 progress_bar = st.progress(0)
# #                 status_text = st.empty()
# #                 batch_results = []
                
# #                 detector = AdvancedDamageDetector()
                
# #                 for idx, file in enumerate(uploaded_files):
# #                     status_text.text(f"Processing {file.name}...")
                    
# #                     # Process image
# #                     file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
# #                     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# #                     annotated_image, detections = detector.detect_damage(image)
                    
# #                     batch_results.append({
# #                         'filename': file.name,
# #                         'detections': detections,
# #                         'image': annotated_image
# #                     })
                    
# #                     progress_bar.progress((idx + 1) / len(uploaded_files))
                
# #                 status_text.text("✅ Batch processing completed!")
                
# #                 # Summary table
# #                 st.subheader("Batch Results Summary")
# #                 summary_data = []
# #                 for result in batch_results:
# #                     scratches = sum(1 for d in result['detections'].values() if 'scratch' in d['class'])
# #                     edge_damages = sum(1 for d in result['detections'].values() if 'edge' in d['class'])
# #                     summary_data.append({
# #                         'Filename': result['filename'],
# #                         'Total Damages': len(result['detections']),
# #                         'Scratches': scratches,
# #                         'Edge Damage': edge_damages
# #                     })
                
# #                 summary_df = pd.DataFrame(summary_data)
# #                 st.dataframe(summary_df, use_container_width=True)
                
# #                 # Download all results
# #                 all_results_json = json.dumps([
# #                     {'filename': r['filename'], 'detections': r['detections']} 
# #                     for r in batch_results
# #                 ], indent=2)
                
# #                 st.download_button(
# #                     label="📦 Download All Results (JSON)",
# #                     data=all_results_json,
# #                     file_name="all_detections.json",
# #                     mime="application/json"
# #                 )
    
# #     # Tab 3: Detection Analytics
# #     with tab3:
# #         st.subheader("Detection Analytics")
        
# #         if not st.session_state.detection_history:
# #             st.info("No detection history yet. Process some images to see analytics.")
# #         else:
# #             # Statistics
# #             total_images = len(st.session_state.detection_history)
# #             total_damages = sum(len(r['detections']) for r in st.session_state.detection_history)
            
# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 st.metric("Total Images Analyzed", total_images)
# #             with col2:
# #                 st.metric("Total Damages Found", total_damages)
# #             with col3:
# #                 avg_damage = total_damages / total_images if total_images > 0 else 0
# #                 st.metric("Average Damages per Image", f"{avg_damage:.1f}")
            
# #             # Damage type distribution
# #             damage_types = {}
# #             for record in st.session_state.detection_history:
# #                 for det in record['detections'].values():
# #                     damage_type = det['class']
# #                     damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
            
# #             if damage_types:
# #                 fig = go.Figure(data=[go.Pie(
# #                     labels=list(damage_types.keys()),
# #                     values=list(damage_types.values()),
# #                     hole=0.4,
# #                     marker_colors=['#ff9800', '#f44336', '#9c27b0']
# #                 )])
# #                 fig.update_layout(title="Damage Type Distribution", height=400)
# #                 st.plotly_chart(fig, use_container_width=True)
            
# #             # Recent detections
# #             st.subheader("Recent Detections")
# #             for record in reversed(st.session_state.detection_history[-5:]):
# #                 with st.expander(f"📸 {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {record['filename']}"):
# #                     col1, col2 = st.columns([1, 1])
# #                     with col1:
# #                         st.image(cv2.cvtColor(record['image'], cv2.COLOR_BGR2RGB), 
# #                                 caption="Detection Result", width=None)
# #                     with col2:
# #                         display_metrics(record['detections'])

# # if __name__ == "__main__":
# #     main()


# # # """
# # # Conveyor Belt Damage Detection System
# # # Streamlit-based Web Application for detecting scratches and edge damage on conveyor belts
# # # """

# # # import streamlit as st
# # # import cv2
# # # import numpy as np
# # # import pandas as pd
# # # from pathlib import Path
# # # import json
# # # import time
# # # from datetime import datetime
# # # import plotly.graph_objects as go
# # # import plotly.express as px
# # # from PIL import Image
# # # import io
# # # import base64
# # # from ultralytics import YOLO
# # # import os
# # # import sys

# # # # Add backend to path
# # # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # # # Page configuration
# # # st.set_page_config(
# # #     page_title="BeltGuard AI - Conveyor Belt Damage Detection",
# # #     page_icon="🛠️",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # Custom CSS
# # # st.markdown("""
# # # <style>
# # #     .main-header {
# # #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# # #         padding: 2rem;
# # #         border-radius: 1rem;
# # #         margin-bottom: 2rem;
# # #         text-align: center;
# # #         color: white;
# # #     }
# # #     .metric-card {
# # #         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
# # #         padding: 1rem;
# # #         border-radius: 1rem;
# # #         text-align: center;
# # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # #     }
# # #     .damage-box {
# # #         padding: 0.5rem;
# # #         border-radius: 0.5rem;
# # #         margin: 0.5rem 0;
# # #     }
# # #     .scratch {
# # #         background-color: #fff3e0;
# # #         border-left: 4px solid #ff9800;
# # #         padding: 0.5rem;
# # #         margin: 0.5rem 0;
# # #     }
# # #     .edge-damage {
# # #         background-color: #ffebee;
# # #         border-left: 4px solid #f44336;
# # #         padding: 0.5rem;
# # #         margin: 0.5rem 0;
# # #     }
# # #     .stButton > button {
# # #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# # #         color: white;
# # #         border: none;
# # #         padding: 0.5rem 2rem;
# # #         border-radius: 0.5rem;
# # #         font-weight: bold;
# # #     }
# # #     .stButton > button:hover {
# # #         transform: translateY(-2px);
# # #         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
# # #         transition: all 0.3s;
# # #     }
# # #     .success-message {
# # #         background-color: #4caf50;
# # #         color: white;
# # #         padding: 1rem;
# # #         border-radius: 0.5rem;
# # #         text-align: center;
# # #     }
# # #     .info-box {
# # #         background-color: #e3f2fd;
# # #         padding: 1rem;
# # #         border-radius: 0.5rem;
# # #         border-left: 4px solid #2196f3;
# # #         margin: 1rem 0;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # Initialize session state
# # # if 'detection_history' not in st.session_state:
# # #     st.session_state.detection_history = []
# # # if 'model_loaded' not in st.session_state:
# # #     st.session_state.model_loaded = False
# # # if 'model' not in st.session_state:
# # #     st.session_state.model = None

# # # class ConveyorBeltDetector:
# # #     """Wrapper class for conveyor belt damage detection"""
    
# # #     def __init__(self, model_path: str = "backend/models/best.pt"):
# # #         self.model_path = Path(model_path)
# # #         self.model = None
# # #         self.class_names = {0: 'scratch', 1: 'edge_damage'}
# # #         self.colors = {
# # #             'scratch': (255, 165, 0),      # Orange for scratches
# # #             'edge_damage': (255, 0, 0)      # Red for edge damage
# # #         }
        
# # #     def load_model(self):
# # #         """Load YOLO model"""
# # #         try:
# # #             if self.model_path.exists():
# # #                 self.model = YOLO(str(self.model_path))
# # #                 return True
# # #             else:
# # #                 # Use default model for demo
# # #                 st.warning("Trained model not found. Using default YOLO model for demonstration.")
# # #                 self.model = YOLO('yolov8n.pt')
# # #                 return True
# # #         except Exception as e:
# # #             st.error(f"Failed to load model: {str(e)}")
# # #             return False
    
# # #     def detect(self, image):
# # #         """
# # #         Run detection on image
        
# # #         Args:
# # #             image: numpy array (BGR format)
            
# # #         Returns:
# # #             annotated_image, detections_dict
# # #         """
# # #         if self.model is None:
# # #             raise ValueError("Model not loaded")
        
# # #         # Convert to RGB for YOLO
# # #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #         height, width = image.shape[:2]
        
# # #         # Run inference
# # #         results = self.model(image_rgb, conf=0.25, iou=0.45)
        
# # #         # Parse detections
# # #         detections = {}
# # #         annotated_image = image_rgb.copy()
        
# # #         for i, result in enumerate(results[0].boxes):
# # #             if result is None:
# # #                 continue
            
# # #             # Get coordinates
# # #             x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
# # #             confidence = float(result.conf[0])
# # #             class_id = int(result.cls[0])
# # #             class_name = self.class_names.get(class_id, 'unknown')
            
# # #             # Store detection
# # #             detections[str(i + 1)] = {
# # #                 "bbox_coordinates": [x1, y1, x2, y2],
# # #                 "class": class_name,
# # #                 "confidence": confidence
# # #             }
            
# # #             # Draw bounding box
# # #             color = self.colors.get(class_name, (0, 255, 0))
# # #             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
# # #             # Add label with background
# # #             label = f"{class_name.upper()}: {confidence:.2f}"
# # #             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
# # #             cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
# # #                          (x1 + label_size[0], y1), color, -1)
# # #             cv2.putText(annotated_image, label, (x1, y1 - 5),
# # #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
# # #         return annotated_image, detections

# # # @st.cache_resource
# # # def load_detector():
# # #     """Load and cache the detector model"""
# # #     detector = ConveyorBeltDetector()
# # #     if detector.load_model():
# # #         st.session_state.model_loaded = True
# # #         st.session_state.model = detector
# # #         return detector
# # #     return None

# # # def display_metrics(detections):
# # #     """Display detection metrics"""
# # #     if not detections:
# # #         return
    
# # #     scratches = sum(1 for d in detections.values() if d['class'] == 'scratch')
# # #     edge_damages = sum(1 for d in detections.values() if d['class'] == 'edge_damage')
    
# # #     col1, col2, col3 = st.columns(3)
    
# # #     with col1:
# # #         st.markdown("""
# # #         <div class="metric-card">
# # #             <h3>Total Damages</h3>
# # #             <h1 style="color: #667eea;">{}</h1>
# # #         </div>
# # #         """.format(len(detections)), unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown("""
# # #         <div class="metric-card">
# # #             <h3>Scratches</h3>
# # #             <h1 style="color: #ff9800;">{}</h1>
# # #         </div>
# # #         """.format(scratches), unsafe_allow_html=True)
    
# # #     with col3:
# # #         st.markdown("""
# # #         <div class="metric-card">
# # #             <h3>Edge Damage</h3>
# # #             <h1 style="color: #f44336;">{}</h1>
# # #         </div>
# # #         """.format(edge_damages), unsafe_allow_html=True)

# # # def create_damage_chart(detections):
# # #     """Create a pie chart of damage types"""
# # #     if not detections:
# # #         return None
    
# # #     damage_counts = {'Scratch': 0, 'Edge Damage': 0}
# # #     for d in detections.values():
# # #         if d['class'] == 'scratch':
# # #             damage_counts['Scratch'] += 1
# # #         else:
# # #             damage_counts['Edge Damage'] += 1
    
# # #     fig = go.Figure(data=[go.Pie(
# # #         labels=list(damage_counts.keys()),
# # #         values=list(damage_counts.values()),
# # #         hole=0.4,
# # #         marker_colors=['#ff9800', '#f44336']
# # #     )])
    
# # #     fig.update_layout(
# # #         title="Damage Distribution",
# # #         showlegend=True,
# # #         height=400,
# # #         margin=dict(t=50, l=0, r=0, b=0)
# # #     )
    
# # #     return fig

# # # def create_confidence_chart(detections):
# # #     """Create a bar chart of confidence scores"""
# # #     if not detections:
# # #         return None
    
# # #     confidences = [d['confidence'] for d in detections.values()]
# # #     damage_types = [d['class'].replace('_', ' ').title() for d in detections.values()]
    
# # #     fig = go.Figure(data=[go.Bar(
# # #         x=damage_types,
# # #         y=confidences,
# # #         marker_color=['#ff9800' if t == 'Scratch' else '#f44336' for t in damage_types],
# # #         text=[f"{c:.2%}" for c in confidences],
# # #         textposition='auto',
# # #     )])
    
# # #     fig.update_layout(
# # #         title="Detection Confidence Scores",
# # #         xaxis_title="Damage Type",
# # #         yaxis_title="Confidence Score",
# # #         yaxis_tickformat='.0%',
# # #         height=400,
# # #         margin=dict(t=50, l=0, r=0, b=0)
# # #     )
    
# # #     return fig

# # # def main():
# # #     # Header
# # #     st.markdown("""
# # #     <div class="main-header">
# # #         <h1>🛠️ BeltGuard AI</h1>
# # #         <p style="font-size: 1.2rem;">Advanced Conveyor Belt Damage Detection System</p>
# # #         <p>Detecting Scratches & Edge Damage with State-of-the-Art AI</p>
# # #     </div>
# # #     """, unsafe_allow_html=True)
    
# # #     # Sidebar
# # #     with st.sidebar:
# # #         st.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=80)
# # #         st.title("BeltGuard AI")
# # #         st.markdown("---")
        
# # #         # Model info
# # #         st.subheader("📊 System Status")
# # #         if st.session_state.model_loaded:
# # #             st.success("✅ Model Loaded")
# # #             st.info("🎯 Model: YOLOv8\n📋 Classes: Scratch, Edge Damage")
# # #         else:
# # #             st.warning("⚠️ Loading Model...")
        
# # #         st.markdown("---")
        
# # #         # Settings
# # #         st.subheader("⚙️ Settings")
# # #         confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
# # #         iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        
# # #         st.markdown("---")
        
# # #         # Info
# # #         st.subheader("ℹ️ About")
# # #         st.markdown("""
# # #         **BeltGuard AI** uses deep learning to detect:
# # #         - 🔍 Surface Scratches
# # #         - ⚠️ Edge Damage
        
# # #         **Features:**
# # #         - Real-time detection
# # #         - Batch processing
# # #         - Export results
# # #         - Detection history
# # #         """)
        
# # #         st.markdown("---")
# # #         st.caption("© 2026 BeltGuard AI | Ripik AI")
    
# # #     # Main content area with tabs
# # #     tab1, tab2, tab3, tab4 = st.tabs([
# # #         "📸 Single Image Detection",
# # #         "📁 Batch Processing",
# # #         "📊 Detection History",
# # #         "📈 Performance Metrics"
# # #     ])
    
# # #     # Tab 1: Single Image Detection
# # #     with tab1:
# # #         col1, col2 = st.columns([1, 1])
        
# # #         with col1:
# # #             st.subheader("Upload Image")
# # #             uploaded_file = st.file_uploader(
# # #                 "Choose a conveyor belt image",
# # #                 type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
# # #                 key="single_upload"
# # #             )
            
# # #             if uploaded_file is not None:
# # #                 # Read image
# # #                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
# # #                 image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# # #                 st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
                
# # #                 # Detect button
# # #                 if st.button("🔍 Detect Damage", key="detect_single"):
# # #                     with st.spinner("Analyzing image..."):
# # #                         detector = st.session_state.model
# # #                         if detector:
# # #                             # Run detection
# # #                             annotated_image, detections = detector.detect(image)
                            
# # #                             # Store in history
# # #                             st.session_state.detection_history.append({
# # #                                 'timestamp': datetime.now(),
# # #                                 'filename': uploaded_file.name,
# # #                                 'detections': detections,
# # #                                 'image': annotated_image
# # #                             })
                            
# # #                             # Display results in second column
# # #                             with col2:
# # #                                 st.subheader("Detection Results")
# # #                                 st.image(annotated_image, caption="Detected Damages", use_column_width=True)
                                
# # #                                 # Metrics
# # #                                 display_metrics(detections)
                                
# # #                                 # Detailed detections
# # #                                 if detections:
# # #                                     st.subheader("📍 Detected Regions")
# # #                                     for det_id, det in detections.items():
# # #                                         x1, y1, x2, y2 = det['bbox_coordinates']
# # #                                         damage_class = det['class'].replace('_', ' ').title()
# # #                                         confidence = det['confidence']
                                        
# # #                                         st.markdown(f"""
# # #                                         <div class="damage-box {det['class']}">
# # #                                             <strong>🔴 Damage {det_id}</strong><br>
# # #                                             Type: {damage_class}<br>
# # #                                             Confidence: {confidence:.2%}<br>
# # #                                             Location: [{x1}, {y1}, {x2}, {y2}]
# # #                                         </div>
# # #                                         """, unsafe_allow_html=True)
                                    
# # #                                     # Export options
# # #                                     col_export1, col_export2 = st.columns(2)
# # #                                     with col_export1:
# # #                                         # Download annotated image
# # #                                         annotated_pil = Image.fromarray(annotated_image)
# # #                                         buf = io.BytesIO()
# # #                                         annotated_pil.save(buf, format="JPEG")
# # #                                         st.download_button(
# # #                                             label="📥 Download Annotated Image",
# # #                                             data=buf.getvalue(),
# # #                                             file_name=f"annotated_{uploaded_file.name}",
# # #                                             mime="image/jpeg"
# # #                                         )
                                    
# # #                                     with col_export2:
# # #                                         # Download JSON
# # #                                         json_str = json.dumps(detections, indent=2)
# # #                                         st.download_button(
# # #                                             label="📄 Download JSON",
# # #                                             data=json_str,
# # #                                             file_name=f"detections_{Path(uploaded_file.name).stem}.json",
# # #                                             mime="application/json"
# # #                                         )
# # #                         else:
# # #                             st.error("Model not loaded. Please check the model file.")
    
# # #     # Tab 2: Batch Processing
# # #     with tab2:
# # #         st.subheader("Batch Image Processing")
# # #         st.markdown('<div class="info-box">Upload multiple images for batch processing. Results will be saved and available for download.</div>', unsafe_allow_html=True)
        
# # #         uploaded_files = st.file_uploader(
# # #             "Choose multiple images",
# # #             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
# # #             accept_multiple_files=True,
# # #             key="batch_upload"
# # #         )
        
# # #         if uploaded_files:
# # #             st.write(f"📁 {len(uploaded_files)} files selected")
            
# # #             if st.button("🚀 Process All Images", key="process_batch"):
# # #                 progress_bar = st.progress(0)
# # #                 status_text = st.empty()
# # #                 batch_results = []
                
# # #                 detector = st.session_state.model
# # #                 if detector:
# # #                     for idx, file in enumerate(uploaded_files):
# # #                         status_text.text(f"Processing {file.name}...")
                        
# # #                         # Read and process image
# # #                         file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
# # #                         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# # #                         annotated_image, detections = detector.detect(image)
                        
# # #                         batch_results.append({
# # #                             'filename': file.name,
# # #                             'detections': detections,
# # #                             'image': annotated_image
# # #                         })
                        
# # #                         # Update progress
# # #                         progress_bar.progress((idx + 1) / len(uploaded_files))
                    
# # #                     status_text.text("✅ Batch processing completed!")
                    
# # #                     # Display results summary
# # #                     st.subheader("Batch Results Summary")
                    
# # #                     # Create summary dataframe
# # #                     summary_data = []
# # #                     for result in batch_results:
# # #                         scratches = sum(1 for d in result['detections'].values() if d['class'] == 'scratch')
# # #                         edge_damages = sum(1 for d in result['detections'].values() if d['class'] == 'edge_damage')
# # #                         summary_data.append({
# # #                             'Filename': result['filename'],
# # #                             'Total Damages': len(result['detections']),
# # #                             'Scratches': scratches,
# # #                             'Edge Damage': edge_damages
# # #                         })
                    
# # #                     summary_df = pd.DataFrame(summary_data)
# # #                     st.dataframe(summary_df, use_container_width=True)
                    
# # #                     # Display each result
# # #                     st.subheader("Individual Results")
# # #                     for result in batch_results:
# # #                         with st.expander(f"📷 {result['filename']}"):
# # #                             col1, col2 = st.columns([1, 1])
# # #                             with col1:
# # #                                 st.image(result['image'], use_column_width=True)
# # #                             with col2:
# # #                                 if result['detections']:
# # #                                     for det_id, det in result['detections'].items():
# # #                                         st.markdown(f"""
# # #                                         <div class="damage-box {det['class']}">
# # #                                             <strong>Damage {det_id}</strong><br>
# # #                                             Type: {det['class'].replace('_', ' ').title()}<br>
# # #                                             Confidence: {det['confidence']:.2%}
# # #                                         </div>
# # #                                         """, unsafe_allow_html=True)
# # #                                 else:
# # #                                     st.success("✅ No damage detected in this image")
                            
# # #                             # Export individual result
# # #                             json_str = json.dumps(result['detections'], indent=2)
# # #                             st.download_button(
# # #                                 label=f"Download {result['filename']} Results",
# # #                                 data=json_str,
# # #                                 file_name=f"detections_{Path(result['filename']).stem}.json",
# # #                                 mime="application/json",
# # #                                 key=f"batch_download_{result['filename']}"
# # #                             )
                    
# # #                     # Batch download all results
# # #                     all_results_json = json.dumps([
# # #                         {'filename': r['filename'], 'detections': r['detections']} 
# # #                         for r in batch_results
# # #                     ], indent=2)
                    
# # #                     st.download_button(
# # #                         label="📦 Download All Results (JSON)",
# # #                         data=all_results_json,
# # #                         file_name="all_detections.json",
# # #                         mime="application/json"
# # #                     )
    
# # #     # Tab 3: Detection History
# # #     with tab3:
# # #         st.subheader("Detection History")
        
# # #         if not st.session_state.detection_history:
# # #             st.info("No detection history yet. Process some images to see history here.")
# # #         else:
# # #             st.write(f"📊 Total analyses: {len(st.session_state.detection_history)}")
            
# # #             # Clear history button
# # #             if st.button("🗑️ Clear History"):
# # #                 st.session_state.detection_history = []
# # #                 st.rerun()
            
# # #             # Display history
# # #             for idx, record in enumerate(reversed(st.session_state.detection_history)):
# # #                 with st.expander(f"📸 {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {record['filename']}"):
# # #                     col1, col2 = st.columns([1, 1])
# # #                     with col1:
# # #                         st.image(record['image'], use_column_width=True)
# # #                     with col2:
# # #                         display_metrics(record['detections'])
                        
# # #                         if record['detections']:
# # #                             st.write("**Detections:**")
# # #                             for det_id, det in record['detections'].items():
# # #                                 st.write(f"- {det['class'].title()}: {det['confidence']:.2%}")
    
# # #     # Tab 4: Performance Metrics
# # #     with tab4:
# # #         st.subheader("Model Performance Metrics")
        
# # #         # Metrics explanation
# # #         st.markdown("""
# # #         <div class="info-box">
# # #             <strong>📊 Evaluation Metric: mF1@0.5-0.95</strong><br>
# # #             The model is evaluated using mean F1 score across IoU thresholds from 0.50 to 0.95 (step 0.05).
# # #             This provides a comprehensive measure of detection accuracy at varying levels of localization strictness.
# # #         </div>
# # #         """, unsafe_allow_html=True)
        
# # #         # Example performance metrics (replace with actual model metrics)
# # #         col1, col2, col3 = st.columns(3)
        
# # #         with col1:
# # #             st.markdown("""
# # #             <div class="metric-card">
# # #                 <h3>mF1@0.5-0.95</h3>
# # #                 <h1 style="color: #4caf50;">0.85</h1>
# # #             </div>
# # #             """, unsafe_allow_html=True)
        
# # #         with col2:
# # #             st.markdown("""
# # #             <div class="metric-card">
# # #                 <h3>Precision</h3>
# # #                 <h1 style="color: #2196f3;">0.88</h1>
# # #             </div>
# # #             """, unsafe_allow_html=True)
        
# # #         with col3:
# # #             st.markdown("""
# # #             <div class="metric-card">
# # #                 <h3>Recall</h3>
# # #                 <h1 style="color: #ff9800;">0.82</h1>
# # #             </div>
# # #             """, unsafe_allow_html=True)
        
# # #         # IoU threshold performance chart
# # #         iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# # #         f1_scores = [0.92, 0.90, 0.88, 0.85, 0.82, 0.78, 0.74, 0.69, 0.63, 0.55]
        
# # #         fig = go.Figure()
# # #         fig.add_trace(go.Scatter(
# # #             x=iou_thresholds,
# # #             y=f1_scores,
# # #             mode='lines+markers',
# # #             name='F1 Score',
# # #             line=dict(color='#667eea', width=3),
# # #             marker=dict(size=10, color='#764ba2')
# # #         ))
        
# # #         fig.update_layout(
# # #             title="F1 Score vs IoU Threshold",
# # #             xaxis_title="IoU Threshold",
# # #             yaxis_title="F1 Score",
# # #             yaxis_range=[0, 1],
# # #             height=400,
# # #             margin=dict(t=50, l=0, r=0, b=0)
# # #         )
        
# # #         st.plotly_chart(fig, use_container_width=True)
        
# # #         # Class-wise performance
# # #         st.subheader("Class-wise Performance")
        
# # #         class_metrics = pd.DataFrame({
# # #             'Class': ['Scratch', 'Edge Damage'],
# # #             'Precision': [0.86, 0.90],
# # #             'Recall': [0.80, 0.84],
# # #             'F1 Score': [0.83, 0.87],
# # #             'AP@0.5': [0.89, 0.92],
# # #             'AP@0.5:0.95': [0.82, 0.88]
# # #         })
        
# # #         st.dataframe(class_metrics, use_container_width=True)
        
# # #         # Confusion matrix placeholder
# # #         st.subheader("Confusion Matrix")
# # #         st.info("Confusion matrix will be available after model evaluation on test dataset.")

# # # # CLI interface function
# # # def run_cli_mode():
# # #     """Run the pipeline in CLI mode"""
# # #     import argparse
    
# # #     parser = argparse.ArgumentParser(description='Conveyor Belt Damage Detection Pipeline')
# # #     parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
# # #     parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
# # #     parser.add_argument('--model_path', type=str, default='backend/models/best.pt', help='Model path')
    
# # #     args = parser.parse_args()
    
# # #     # Import pipeline module
# # #     from backend.pipeline import ConveyorBeltDamageDetector
    
# # #     detector = ConveyorBeltDamageDetector(args.model_path)
# # #     detector.process_directory(args.image_dir, args.output_dir)

# # # if __name__ == "__main__":
# # #     import sys
    
# # #     # Check if running in CLI mode
# # #     if len(sys.argv) > 1 and sys.argv[1] in ['--image_dir', '--help']:
# # #         run_cli_mode()
# # #     else:
# # #         # Load detector
# # #         detector = load_detector()
# # #         if detector:
# # #             main()
# # #         else:
# # #             st.error("Failed to initialize the detection system. Please check the model file.")
# # #             st.info("You can train a model using: python backend/train.py --dataset_dir /path/to/dataset")