# NAYRA — Setup & Installation Guide

## 1. Prerequisites

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS (x86_64 / Apple Silicon).
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.12 recommended).
- **Hardware**:
  - Minimum: Intel Core i5 / AMD Ryzen 5, 8 GB RAM.
  - Recommended: Intel Core i7 / AMD Ryzen 7, 16 GB RAM, NVIDIA RTX GPU (for hardware acceleration).
- **Peripherals**:
  - Web camera or Android IP Webcam stream (HTTP/RTSP).
  - Microphone and speaker/headphones.

---

## 2. Environment Setup

### 2.1 Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2.2 Dependency Installation
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Model Files Setup

Place required offline model checkpoints in the `models/` directory:
- `models/yolov10n.onnx` — Lightweight YOLOv10 object detector.
- `models/tts/kokoro/kokoro-v1.0.onnx` and `models/tts/kokoro/voices-v1.0.bin` — Kokoro-82M TTS neural synthesis engine.
- `models/stt/whisper/models--Systran--faster-whisper-tiny/` — Faster-Whisper offline STT model.
- `models/depth_anything_v2_vits.onnx` or `models/midas_v21_small.onnx` — Monocular depth estimation (Module 05).
- `models/face_detection_yunet_2023mar.onnx` and `models/face_recognition_sface_2021dec.onnx` — Face detection & embeddings (Module 06).
- `models/ocr/ch_PP-OCRv4_det.onnx` and `models/ocr/devanagari_PP-OCRv5_mobile_rec.onnx` — On-demand text extraction (Module 07).

*Note: All inference runs locally on CPU via ONNX Runtime and CTranslate2 with zero cloud dependencies.*

---

## 4. Launching NAYRA

### 4.1 Standard Full Assistant Mode (Webcam + Voice)
```bash
python main.py --camera 0 --lang hi
```

### 4.2 Android Phone Camera Streaming Mode
To stream from an Android phone running the IP Webcam app:
```bash
python main.py --camera http://192.168.0.101:8080/video --lang hi
```
*(Replace `192.168.0.101:8080` with the actual IP address displayed in your phone's IP Webcam app).*

### 4.3 Interactive Text-Voice Console (Development & Testing)
```bash
python main.py --no-camera --text-debug --lang hi
```
This mode lets you interactively test queries in Hindi or English:
- `"Main kahan hoon?"`
- `"Kya rasta saaf hai?"`
- `"Mujhe 10 minute baad dawai lene ki yaad dilana"`
- `"Mera phone kahan hai?"`
- `"Madad karo emergency hai"`

### 4.4 Automated System Verification Smoke Test
```bash
python main.py --smoke-test --no-camera
```

---

## 5. Running Test Suites

Run the full verified pytest regression suite across all 18 modules (208 tests):
```bash
python -m pytest 11_obstacle_awareness/tests/ 12_scene_understanding/tests/ 13_alzheimer_assistance/tests/ 14_daily_assistance/tests/ 15_safety_sos/tests/ 16_voice_interaction/tests/ 17_assistant_orchestrator/tests/ 18_application/tests/ tests/ 10_object_memory/tests/ -v
```
