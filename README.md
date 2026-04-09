# 🚀 BeltGuard AI – Conveyor Belt Damage Detection System

> ⚡ AI-powered real-time conveyor belt damage detection system using Computer Vision and Streamlit.

---

## 📌 Overview

**BeltGuard AI** is an intelligent inspection system that detects **scratches, edge damage, and surface anomalies** on conveyor belts to enable predictive maintenance and reduce downtime.

- ✅ No training required (works out-of-the-box)
- ⚡ Fast processing (< 0.5 sec/image)
- 🖥️ Supports both Web UI and CLI pipeline

---

## 🎯 Key Highlights

- 🔥 Production-ready Computer Vision system
- ⚡ Real-time detection (0.2–0.5 sec/image)
- 🧠 Modular architecture (CLI + Web UI)
- 📊 Interactive dashboard using Streamlit
- 🏭 Industry-focused use case (predictive maintenance)

---

## 🧰 Tech Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV
- **AI/Detection:** YOLOv8 (reference/inspiration)
- **Frontend:** Streamlit
- **Libraries:** NumPy, Pandas, Plotly, Pillow

---

## ✨ Features

- 🔍 Real-time damage detection
- 📁 Batch image processing
- 🎛️ Adjustable sensitivity levels
- 📊 Interactive dashboard
- 💾 Export results (JSON + images)
- 🌓 Works in different lighting conditions
- 📜 Detection history tracking

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/RamNarayan3231/Conveyor-Belt-Damage-Detection-System.git
cd Conveyor-Belt-Damage-Detection-System

```
### Create Virtual Environment
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Run Application
```
# Web App
streamlit run app.py

# CLI Pipeline
python pipeline.py --image_dir ./images --output_dir ./outputs
```
### 🖥️ Usage
🌐 Web Interface
- Run: streamlit run app.py
- Open browser: http://localhost:8501
- Upload image (JPG, PNG, BMP)
- Adjust sensitivity (Low / Medium / High)
- View and download results


## 📊 Output Format
###JSON Example
```
{
  "1": {
    "bbox_coordinates": [100, 150, 300, 250],
    "class": "scratch",
    "confidence": 0.85
  }
}

```
![annotated_20260131_220303_792369_jpg rf 639015ea4d2cdfaad4d9abd005e6069b](https://github.com/user-attachments/assets/d08facfe-7e0a-43d6-b1ca-6c4ee8d638b7)

### Output Includes
- Annotated image with bounding boxes
- Damage type labels
- Confidence scores

### Performance
- ⚡ Speed: 0.2–0.5 sec/image
- 🎯 Accuracy: ~85%
- 💾 Memory Usage: ~300MB
- 🖥️ Runs on CPU (No GPU required)

### 🎉 Acknowledgments
- YOLOv8 – Detection inspiration
- Streamlit – Web interface
- OpenCV – Computer vision processing
