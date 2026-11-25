# NAYRA — Complete Module Architecture Documentation (Modules 01–18)

This document provides technical documentation for all eighteen modules composing the unified NAYRA Assistive AI Companion.

---

## 1. Perception Subsystems (Modules 01–07)

### Module 01: Camera Input (`01_camera_input/`)
- **Purpose**: Provides low-latency, thread-isolated video capture from local webcams, RTSP streams, and Android IP Webcam streams.
- **Key Classes**: `CameraCapture`, `VideoBuffer`.
- **Inputs**: Device index or network URL.
- **Outputs**: Fresh OpenCV BGR frames via non-blocking ring buffer.

### Module 02: Camera Manager (`02_camera_manager/`)
- **Purpose**: Lifecycle management, automatic reconnect on frame drops, resolution negotiation, and frame rate throttling.
- **Key Classes**: `CameraManager`, `CameraStatus`.
- **Inputs**: `CameraSettings`.
- **Outputs**: Managed frame streams with health telemetry.

### Module 03: Object Detection (`03_object_detection/`)
- **Purpose**: Real-time multi-class object detection using YOLOv10n ONNX.
- **Key Classes**: `ObjectDetector`, `DetectionResult`.
- **Inputs**: RGB/BGR frame array.
- **Outputs**: Bounding boxes, class names, confidence scores.

### Module 04: Object Tracking (`04_object_tracking/`)
- **Purpose**: Persistent multi-object association across frames using ByteTrack.
- **Key Classes**: `ByteTracker`, `TrackState`.
- **Inputs**: Frame detections from Module 03.
- **Outputs**: Consistent Track IDs and velocity vectors.

### Module 05: Spatial Awareness (`05_spatial_awareness/`)
- **Purpose**: Monocular depth estimation, horizontal zoning (`LEFT`, `CENTER`, `RIGHT`), and vertical proximity categorisation.
- **Key Classes**: `DepthEstimator`, `SpatialAnalyzer`, `SpatialZone`.
- **Inputs**: BGR frames and bounding boxes.
- **Outputs**: Spatial coordinates relative to camera viewport.

### Module 06: Person Recognition (`06_person_recognition/`)
- **Purpose**: Facial detection, feature embedding generation, and cosine similarity lookup against known family/caregiver faces.
- **Key Classes**: `FaceDatabase`, `PersonRecognitionPipeline`.
- **Inputs**: BGR frame crops.
- **Outputs**: Identified person identity, confidence, and relationship metadata.

### Module 07: OCR & Read Aloud (`07_ocr/`)
- **Purpose**: Optical character recognition for scene text, medicine labels, document reading, and signage.
- **Key Classes**: `AsyncOCRWorker`, `OCREngine`, `OCRResult`.
- **Inputs**: Frame regions of interest.
- **Outputs**: Extracted text strings and reading-order layout.

---

## 2. Voice & Speech Subsystems (Modules 08–09)

### Module 08: Text-to-Speech (`08_tts/`)
- **Purpose**: High quality, local, offline neural voice synthesis using Kokoro-82M.
- **Key Classes**: `TTSManager`, `KokoroEngine`, `AudioPlayer`.
- **Inputs**: Spoken text, language code (`hi` / `en`).
- **Outputs**: Synthesized PCM audio waveforms played via local speakers.

### Module 09: Voice Commands & STT (`09_voice/`)
- **Purpose**: Offline speech recognition via Whisper / Faster-Whisper, microphone stream ingestion, and command phrase matching.
- **Key Classes**: `VoiceCommandRecognizer`, `PhoneAudioReceiver`, `CommandMatcher`.
- **Inputs**: Audio PCM buffer (16 kHz).
- **Outputs**: Transcribed text strings with language tags.

---

## 3. Cognitive & Spatial Memory (Module 10)

### Module 10: Object Location Memory (`10_object_memory/`)
- **Purpose**: Persistent episodic memory of where objects were last seen in physical space relative to furniture anchors.
- **Key Classes**: `ObjectMemoryManager`, `ObservationFilter`, `VoiceGuidedObjectFinder`.
- **Storage**: SQLite database (`data/object_memory.db`).
- **Voice Response**: Natural answers ("Aapka phone saamne table par hai").

---

## 4. Higher-Level Assistive Intelligence (Modules 11–18)

### Module 11: Obstacle Awareness & Guidance (`11_obstacle_awareness/`)
- **Purpose**: Real-time path hazard detection, collision risk evaluation, and spoken directional guidance.
- **Key Classes**: `ObstacleDetector`, `ObstacleWarningGenerator`, `ObstacleService`.
- **Priority**: Emits alerts at `SpeechPriority.CRITICAL_SAFETY` or `OBSTACLE_WARNING`.
- **Voice Output**: e.g., *"Rukye! Saamne kursi hai"* or *"Aapke saamne ka raasta bilkul saaf hai"*.

### Module 12: Contextual Scene Understanding (`12_scene_understanding/`)
- **Purpose**: Multi-modal scene synthesis combining objects, people count, spatial relationships, and OCR text into natural verbal descriptions.
- **Key Classes**: `SceneAnalyzer`, `SceneDescription`, `SceneService`.
- **Voice Output**: e.g., *"Aapke saamne ek vyakti khade hain aur table par laptop rakha hai"*.

### Module 13: Alzheimer's & Cognitive Assistance (`13_alzheimer_assistance/`)
- **Purpose**: Reassuring temporal/spatial orientation (day, date, time, location), familiar person recall, and calming repetition support.
- **Key Classes**: `PatientProfileManager`, `OrientationGuide`, `AlzheimerService`.
- **Voice Output**: e.g., *"Aap bilkul surakshit hain aur apne ghar par hain. Aaj Somwaar, 23 November hai"*.

### Module 14: Daily Assistance & Reminders (`14_daily_assistance/`)
- **Purpose**: Medication and hydration reminder scheduling, periodic background checking, voice snooze and dismissal.
- **Key Classes**: `ReminderDB`, `ReminderScheduler`, `DailyService`.
- **Storage**: SQLite database (`data/reminders.db`).
- **Voice Output**: e.g., *"Dhyan dijiye! Aapka reminder hai: Dawai lene ka samay"*.

### Module 15: Safety & SOS Assistance (`15_safety_sos/`)
- **Purpose**: Voice-triggered SOS, 2-stage verification countdown, local emergency event logging, and caregiver alert dispatch.
- **Key Classes**: `SOSManager`, `SOSState`, `SafetyService`.
- **Storage**: `data/emergency_contacts.json`, `data/emergency_alerts.jsonl`.
- **Voice Output**: Preempts lower-priority speech immediately with *"Kya aapko sach mein emergency madad chahiye?"*.

### Module 16: Shared Voice Interaction Layer (`16_voice_interaction/`)
- **Purpose**: System-wide priority speech queue, preemption manager, and acoustic feedback shield.
- **Key Classes**: `PrioritySpeechQueue`, `AcousticShield`, `VoiceManager`.
- **Guarantee**: Sub-50ms priority arbitration; acoustic shield blocks mic echo during playback + 1.2s cooldown.

### Module 17: Assistant Orchestrator (`17_assistant_orchestrator/`)
- **Purpose**: Central brain; classifies multilingual spoken queries, tracks dialog context, and routes to appropriate domain modules.
- **Key Classes**: `IntentClassifier`, `AssistantOrchestrator`.

### Module 18: Unified Application (`18_application/` & `main.py`)
- **Purpose**: Topological startup sequence, perception loop, voice loop, signal handling, and clean shutdown.
- **Key Classes**: `NayraApplication`, `AppConfig`.
