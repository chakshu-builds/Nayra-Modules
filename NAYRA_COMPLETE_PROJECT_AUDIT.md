# NAYRA — Complete Existing Repository Audit
**Document:** `NAYRA_COMPLETE_PROJECT_AUDIT.md`  
**System Version:** NAYRA MVP 1.0  
**Audit Date:** 2026-11-23  
**Status:** Comprehensive Baseline Established  

---

## 1. Executive Summary

NAYRA is an offline-first, local-first intelligent assistive companion designed for visually impaired individuals, elderly users, and individuals coping with Alzheimer's disease or cognitive impairment.

This audit evaluates all existing components (Modules 01–10, configurations, models, test suites, and root-level scripts) to establish a rigorous baseline prior to the implementation of the remaining assistive modules (Modules 11–18) and the central voice-first orchestration layer.

### Key Audit Findings
1. **Modules 01–09 are functional, tested, and strictly FROZEN:**
   - **01 (Camera Input) & 02 (Camera Manager):** Industrial MJPEG stream receiver with 1-frame zero-lag buffer and automatic reconnect.
   - **03 (Object Detection):** YOLOv10n ONNX model on CPU/CUDA with COCO 80 classes.
   - **04 (Object Tracking):** ByteTrack with 2-stage association and Kalman filtering.
   - **05 (Spatial Awareness):** MiDaS v2.1 Small & Depth Anything V2 with inner-ROI median disparity sampling and 3-zone horizontal division.
   - **06 (Person Identification):** YuNet face detection + SFace cosine similarity face matching + SQLite context database.
   - **07 (OCR):** DBNet text detector + CRNN / SVTR recognition with Devanagari script routing.
   - **08 (TTS):** Kokoro-82M offline TTS (24 kHz float32) with dual PortAudio (laptop) and NetworkAudioPlayer (phone) outputs.
   - **09 (Voice Recognition):** Offline Whisper-tiny (int8) with Hindi/Hinglish/English support and energy-based VAD.
2. **Module 10 (Object Location Memory) is functional & verified:**
   - SQLite observation database (`data/memory/object_memory.db`) with horizontal zone and anchor furniture relational mapping.
   - Thread-safe live observation cache decoupling real-time visual tracking from database write deduplication.
3. **Core Architectural Gaps Identified:**
   - **No Shared Voice-First Interaction Bus:** Modules 07, 09, and 10 maintain independent audio loops, risking PortAudio/socket resource contention.
   - **No Central Intent Orchestrator:** User currently has to run separate scripts (`run_object_memory.py`, `run_read_aloud.py`, `main.py`).
   - **Missing Assistive Domains:** Obstacle Direction Guidance (Module 11), Contextual Scene Description (Module 12), Alzheimer's Care & Familiar Person Recall (Module 13), Medication & Daily Reminders (Module 14), and Emergency SOS (Module 15).
   - **Lack of Unified Event / Schema Layer:** Modules communicate via custom ad-hoc dictionaries or direct class imports rather than standardized shared interfaces.

---

## 2. Comprehensive Module Audit & Inventory

| Module | Purpose | Status | Strengths | Gaps & Integration Needs |
|---|---|---|---|---|
| **01_camera_input** | Threaded RTSP/HTTP MJPEG stream capture | **Complete (Frozen)** | Zero-lag buffer, auto-reconnect, 35 FPS capability. | Single consumer model; needs shared feed distribution. |
| **02_camera_manager** | Facade over camera sources and status | **Complete (Frozen)** | Clean abstraction over OpenCV and MJPEG clients. | Ready for central orchestrator lifecycle control. |
| **03_object_detection** | YOLOv10n ONNX detection (COCO 80) | **Complete (Frozen)** | Fast inference (~9.8ms GPU, ~45ms CPU), letterbox preprocessing. | Confidence threshold tuned to 0.30 for assistive sensitivity. |
| **04_object_tracking** | ByteTrack Kalman filter tracking | **Complete (Frozen)** | Sub-millisecond CPU tracking, trajectory direction estimation. | Returns confirmed tracks with 0 missed frames. |
| **05_spatial_awareness** | Monocular depth & proximity | **Complete (Frozen)** | Inner-ROI median depth sampling, LEFT/CENTER/RIGHT zones. | Foundation for Module 11 Obstacle Awareness. |
| **06_person_recognition** | YuNet + SFace familiar person identification | **Complete (Frozen)** | Consent-based face DB, relationship metadata, privacy compliant. | Needs voice-driven inquiry ("Kaun hai saamne?"). |
| **07_ocr** | Multilingual bilingual OCR (Hindi/English) | **Complete (Frozen)** | DBNet + SVTR, Shirorekha detection, reading order. | Needs unified voice-trigger ("Ye page padhkar sunao"). |
| **08_tts** | Kokoro-82M offline speech synthesis | **Complete (Frozen)** | 24 kHz high-quality audio, phoneme normalization, audio queue. | Needs centralized priority queue (Safety > User > Info). |
| **09_voice** | Whisper-tiny offline speech recognition | **Complete (Frozen)** | Fast int8 inference, Hindi/Hinglish prompt biasing, VAD. | Needs single shared listener instance for all intents. |
| **10_object_memory** | Spatial object location memory | **Complete (Active)** | SQLite persistence, relational anchoring, 5-tier visibility check. | Needs extension into Alzheimer's cognitive care. |
| **11_obstacle_awareness** | Directional obstacle hazard guidance | **Missing** | Depth & tracking exist in Module 05. | Must generate timely spoken alerts with repetition suppression. |
| **12_scene_understanding** | Natural environment description | **Missing** | YOLO + Depth + OCR + Person data available. | Must synthesize scene into concise natural language sentences. |
| **13_alzheimer_assistance** | Cognitive memory & person reminder | **Missing** | Modules 06 and 10 provide underlying data. | Caregiver profiles, familiar person prompts, memory logs. |
| **14_daily_assistance** | Medication and schedule reminders | **Missing** | None. | SQLite schedule DB, background timer, voice repeat/snooze. |
| **15_safety_sos** | Voice-triggered emergency SOS | **Missing** | None. | Confirmation gate, emergency contact store, SMS payload. |
| **16_voice_interaction** | Shared voice interaction layer | **Missing** | Modules 08 and 09 exist independently. | Central priority speech queue, acoustic feedback shield. |
| **17_assistant_orchestrator** | Central deterministic intent router | **Missing** | Ad-hoc matchers exist in 09 and 10. | Unified NLP router for all domains without paid cloud LLMs. |
| **18_application** | Main unified startup & lifecycle | **Partial (`main.py`)** | `main.py` only runs camera/vision preview. | Needs unified voice-first assistant loop. |

---

## 3. Disconnected Functionality & Redundancies

1. **Independent TTS Instantiations:**  
   Both `09_voice/read_aloud_pipeline.py` and `10_object_memory/voice_guidance.py` instantiate their own `TTSManager`. When run together, they compete for PortAudio speaker resources.  
   *Resolution:* Module 16 must provide a singleton `SharedVoiceInterface` with a thread-safe, priority-based speech queue.
2. **Duplicate Query Parsing:**  
   `09_voice/command_matcher.py` handles "read" commands; `10_object_memory/nlp/query_parser.py` handles "where" commands.  
   *Resolution:* Module 17 will unify all domain queries into a hierarchical intent routing engine.
3. **Redundant Camera Consumers:**  
   If multiple background threads instantiate `LiveCameraStream` or `CameraManager` on the same URL, network bandwidth and HTTP sockets are duplicated.  
   *Resolution:* The camera feed must be managed as a single broadcast stream delivering frames to vision workers.

---

## 4. Hardware Dependencies & Operational Feasibility

- **Host Machine:** Windows 10/11 x64, Python 3.12, NVIDIA RTX 4050 Laptop GPU (or pure CPU fallback).
- **Physical Sensors:**
  - Android Smartphone running IP Webcam (`/video` MJPEG stream and `/audio.wav` microphone stream over local Wi-Fi).
  - Alternative: USB Webcam (index 0) + Laptop Microphone.
- **Offline / Local Guarantee:**
  - All neural network models (`yolov10n.onnx`, `midas_v21_small.onnx`, `yunet.onnx`, `sface.onnx`, `ch_PP-OCRv4`, `whisper-tiny-int8`, `kokoro-v0_19.onnx`) reside in `models/`.
  - Zero cloud calls, zero external API keys, zero paid services.
