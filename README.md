# Fire & Smoke Detection Web Application

Detecting Fire and Smoke in real-time using Computer Vision, PyTorch, and Flask.

Early fire and smoke detection plays a critical role in saving lives, reducing property damage, and minimizing operational downtime. This project provides an end-to-end deep learning pipeline and an interactive web surveillance application for automated hazard detection across images, video files, and live webcam feeds.

---

## Demo & Sample Outputs
- **Demo GIF**: `utils/demo.gif`
- **Model Training Accuracy & Loss**: `utils/accuracy.png` and `utils/trainloss.png`
- **Sample Results**: `utils/fire.png`, `utils/smoke.png`, `utils/neutral.png`

---

## Key Features

1. **Interactive Web Dashboard**:
   - Modern cybersecurity/surveillance dark-mode UI.
   - Real-time hazard alert banner (🔴 Fire, 🟠 Smoke, 🟢 Neutral).
   - Audio siren warning (with mute/unmute toggle).
   - Class confidence percentage meters and probability distribution charts.

2. **Multi-Input Inference**:
   - **Image Detection**: Drag & drop custom images or click one-click samples from the dataset gallery.
   - **Video Analysis**: Upload MP4/AVI videos for frame-by-frame analysis, hazard timeline logging, and detection breakdown.
   - **Live Webcam Surveillance**: Direct browser camera access with real-time HUD and FPS counter.

3. **Production REST API**:
   - `GET  /api/health` - Check model status, device, and PyTorch version.
   - `POST /api/predict/image` - Classify image (multipart upload or base64 JSON).
   - `POST /api/predict/frame` - High-speed base64 streaming endpoint for live cameras.
   - `POST /api/predict/video` - Video processing with chronological detection timeline.
   - `GET  /api/samples` - Retrieve sample test images from `test-imgs/`.

---

## Model Architecture
- **Backbone**: Pretrained `ResNet-50` (Transfer Learning on ImageNet).
- **Classifier Head**:
  $$\text{Linear}(2048 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 3) \to \text{Softmax}$$
- **Classes**: `['Fire', 'Neutral', 'Smoke']`
- **Validation Accuracy**: ~93%

---

## Project Structure

```
Fire-Smoke-Detection/
├── trained-models/
│   └── model_final.pth          # PyTorch ResNet-50 trained weights
├── test-imgs/                   # Sample evaluation images
├── utils/                       # Performance plots and demo images
│
├── model_service.py             # Inference engine (CPU/CUDA, pre-processing, video processing)
├── app.py                       # Flask web server & REST API controller
├── run.py                       # CLI launcher script
├── requirements.txt             # Project dependencies
│
├── templates/
│   └── index.html               # Web surveillance dashboard
├── static/
│   ├── css/
│   │   └── style.css            # Responsive dark-theme dashboard stylesheet
│   └── js/
│       └── main.js              # Frontend UI controller, audio alarm, and WebRTC streaming
│
├── tests/
│   └── test_api.py              # Automated unit and API integration tests
│
├── Training.ipynb               # Original model training notebook
├── Inference.ipynb              # Original notebook inference experiments
└── README.md                    # Project documentation
```

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10 - 3.13 supported)
- (Optional) NVIDIA GPU with CUDA for accelerated inference (CPU is automatically supported)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python run.py
```
Or with custom port/host:
```bash
python run.py --host 0.0.0.0 --port 5000
```

Open your browser and navigate to:
```
http://localhost:5000/
```

---

## Running Automated Tests

Run the full automated test suite (verifying model loading, predictions on sample images, and API endpoints):

```bash
pytest tests/test_api.py -v
```

---

## REST API Usage Examples

### Health Check
```bash
curl -X GET http://localhost:5000/api/health
```

### Predict Image (File Upload)
```bash
curl -X POST -F "file=@test-imgs/26.jpg" http://localhost:5000/api/predict/image
```
**Response:**
```json
{
  "prediction": "Fire",
  "confidence": 100.0,
  "is_hazard": true,
  "hazard_level": "CRITICAL",
  "color_hex": "#EF4444",
  "probabilities": {
    "Fire": 100.0,
    "Neutral": 0.0,
    "Smoke": 0.0
  },
  "latency_ms": 93.3
}
```

### Predict Sample by Name
```bash
curl -X POST http://localhost:5000/api/predict/image \
  -H "Content-Type: application/json" \
  -d '{"sample_filename": "image_0.jpg"}'
```

---

## Dataset
- [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip)
- Contains 1,000 images per class across Train and Test splits (`Fire`, `Neutral`, `Smoke`).

---

## References
1. PyImageSearch - [Fire and Smoke Detection with Deep Learning](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/)
2. DeepQuestAI - [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset)
