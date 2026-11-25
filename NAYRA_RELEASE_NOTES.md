# NAYRA — Release Notes (v1.0-RC1 MVP)

**Release Version:** v1.0-RC1 (Release Candidate 1)  
**Release Date:** November 24, 2026  
**Target Architecture:** Windows 10/11 Laptop + Android Smartphone (Local Wi-Fi Subnet)  
**Inference Modality:** 100% Local Offline Neural Inference (CPU / ONNX Runtime / CTranslate2)  
**Voice Modality:** Central Voice-First Interaction (Kokoro-82M TTS + Faster-Whisper STT)  

---

## 1. Release Overview

NAYRA v1.0-RC1 is the first comprehensive Release Candidate of the NAYRA modular assistive companion. It integrates perception, episodic memory, spatial reasoning, cognitive grounding, daily assistance, two-stage safety SOS, and centralized voice interaction into a single cohesive runtime.

All 18 architectural modules are integrated and validated via an automated test suite of **208 passed tests (100% pass rate)**.

---

## 2. Core Functional Highlights

### 2.1 Voice-First Central Interaction
- **Shared Priority Queue:** Thread-safe heap priority scheduling (`CRITICAL_SAFETY` > `EMERGENCY_INTERACTION` > `OBSTACLE_WARNING` > `USER_QUERY_RESPONSE` > `REMINDER` > `INFORMATIONAL`).
- **Immediate Preemption (< 100 ms):** Incoming critical safety alerts actively interrupt lower-priority active audio playback via `_preempt_event` and `tts_backend.stop()`.
- **Acoustic Feedback Shield:** Automatically gates microphone ingestion during TTS playback plus a 1.2-second post-speech cooldown to eliminate Whisper self-triggering loops.
- **Multilingual Intent Classifier:** Recognizes Hindi, Hinglish, and English voice queries with keyword safety protection against extreme lengths and punctuation attacks.

### 2.2 Assistive & Cognitive Intelligence
- **Object Location Memory (Module 10):** 4 truthful visibility states (`currently_visible`, `recently_observed`, `previously_observed`, `never_detected`/`unsupported`). Zero hallucination of physical coordinates.
- **Obstacle Awareness (Module 11):** Real-time horizontal zoning (`LEFT`, `CENTER`, `RIGHT`) with repetition suppression and mutex-protected cache pruning.
- **Scene Understanding (Module 12):** Multi-modal synthesis combining object counts, spatial layouts, and active OCR text into cohesive spoken sentences.
- **Alzheimer's Care (Module 13):** Calming temporal grounding (day, date, time) and familiar person recall without false identity fallback.
- **Daily Reminders (Module 14):** SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`), automatic corruption recovery with spoken alerts, duplicate prevention, and explicit snooze/completion commands.
- **Two-Stage Safety SOS (Module 15):** Voice-triggered emergency verification with 10-second countdown, spoken cancellation, and structured audit logging (`data/emergency_alerts.jsonl`).

---

## 3. Engineering Quality & Reliability Upgrades

1. **SQLite WAL & Rollback:** Concurrency lock contention resolved between background schedulers and voice query threads; transactions automatically rollback on unhandled errors.
2. **Transparent Corruption Auto-Recovery:** Corrupt database headers trigger automatic backup (`.corrupt.<timestamp>.bak`), clean schema rebuild, and an explicit spoken caregiver alert rather than silent data deletion.
3. **Idempotent Lifecycle:** Application initialization, startup, and shutdown can be executed repeatedly across multiple cycles with zero thread leaks.
4. **Reproducible Python Manifest:** `requirements.txt` updated to include all core neural and speech dependencies (`kokoro-onnx`, `soundfile`, `sounddevice`, `faster-whisper`).
5. **Zero Cloud Telemetry:** 100% offline verification; socket interceptor asserts zero non-loopback outbound internet connections.

---

## 4. Known Limitations & Constraints

- **Wearable Hardware Pending:** Designed for future wearable smart glasses; currently operates via laptop webcam or Android smartphone IP Webcam over local Wi-Fi.
- **Physical Cellular Dispatch:** Emergency SOS writes to local disk logs (`data/emergency_alerts.jsonl`); automated cellular SMS/call dispatch is not yet enabled.
- **Cold Startup Latency:** Cold model loading takes **~4.17 s** on CPU (YOLO: 106 ms, Kokoro: 1.10 s, Whisper: 2.96 s).
- **Process Memory Footprint:** Windows Process Working Set (RSS) with full neural weights resident in RAM is **~182 MB to 858 MB**.

---

## 5. Verification Commands

```bash
# Full automated regression suite (208 tests)
python -m pytest 11_obstacle_awareness/tests/ 12_scene_understanding/tests/ 13_alzheimer_assistance/tests/ 14_daily_assistance/tests/ 15_safety_sos/tests/ 16_voice_interaction/tests/ 17_assistant_orchestrator/tests/ 18_application/tests/ tests/ 10_object_memory/tests/ -q

# System smoke test
python main.py --smoke-test --no-camera

# Interactive text-voice console
python main.py --no-camera --text-debug --lang hi
```
