# NAYRA — Features and Limitations Specification

## 1. Executive Summary

NAYRA is an offline-first, voice-guided AI assistive companion designed to empower:
1. **Visually Impaired Individuals** through spatial orientation, obstacle hazard warnings, contextual scene descriptions, text reading, and object location memory.
2. **Elderly Individuals** through conversational support, scheduled medication reminders, hydration prompts, and reliable safety assistance.
3. **Individuals Experiencing Alzheimer's Disease or Cognitive Decline** through gentle temporal/spatial grounding, familiar face recognition, and calming reassurance.

NAYRA operates completely locally without requiring cloud APIs or recurring subscriptions.

---

## 2. Core Feature Matrix & Target User Personas

| Feature | Target Persona | Module | Spoken Language | Output Modality |
| :--- | :--- | :--- | :--- | :--- |
| **Object Location Memory** | Visually Impaired, Alzheimer's | Module 10 | Hindi, Hinglish, English | Spoken audio guidance + spatial anchors |
| **Obstacle Awareness & Path Check** | Visually Impaired, Elderly | Module 11 | Hindi, Hinglish, English | Priority audio hazard alerts ("Rukye! Saamne chair hai") |
| **Contextual Scene Understanding** | Visually Impaired | Module 12 | Hindi, English | Spoken scene narration ("Saamne laptop aur table hai") |
| **Cognitive Grounding & Orientation** | Alzheimer's, Dementia | Module 13 | Hindi, English | Calming date, time, and home reassurance |
| **Familiar Person Identification** | Alzheimer's, Visually Impaired | Module 13 & 06 | Hindi, English | Name, relationship, and caregiver notes |
| **Medication & Routine Reminders** | Elderly, Alzheimer's | Module 14 | Hindi, English | Spoken scheduled alerts, snooze, dismiss |
| **Two-Stage Safety & SOS** | Elderly, All Users | Module 15 | Hindi, English | Preemptive voice confirmation, emergency dispatch |
| **Read Aloud (OCR)** | Visually Impaired, Elderly | Module 07 | Hindi, English | On-demand text reading from camera view |

---

## 3. Detailed Operational Capabilities

### 3.1 Voice-First Interaction System
- All system output is routed through a central priority queue:
  1. `CRITICAL_SAFETY`: Immediate hazard warnings (e.g. collision trajectory).
  2. `EMERGENCY_INTERACTION`: Two-stage SOS confirmation dialogue.
  3. `OBSTACLE_WARNING`: Directional obstacle alerts (left, ahead, right).
  4. `USER_QUERY_RESPONSE`: Answers to direct user speech commands.
  5. `REMINDER`: Medication, hydration, and scheduled appointments.
  6. `INFORMATIONAL`: Scene overviews and startup greetings.
- **Acoustic Feedback Shield**: Prevents audio echo loop into Whisper STT by muting speech recognition during audio playback + a 1.2-second post-speech cooldown.
- **Cognitive Truthfulness & Honest Reassurance**: Never fabricates object locations, never invents caregiver names or arrival times, never claims everything is fine or safe without verified proof, and avoids unverified safety assertions.
- **Obstacle Perception Truthfulness**: An inactive camera, missing frames, or stale perception timestamps (> 5s) are NEVER interpreted as a clear path. NAYRA explicitly alerts the user that perception is inactive or stale.
- **Medical & Reminder Guardrails**: Spoken reminders never automatically mark medication as taken without explicit user confirmation. Tests and background verifications use isolated databases to prevent polluting user schedules.
- **Emergency Disclaimer**: Local JSONL emergency logs record event data locally and prompt contact with designated caregivers; they are not a cellular or 911 dispatch integration. Cancelling an SOS alert confirms cancellation without claiming the user is unconditionally safe.

### 3.2 Spatial Perception & Grounding
- Object detection runs using YOLOv10n on ONNX Runtime (CPU or CUDA).
- Monocular spatial zoning splits camera field-of-view into horizontal zones (`LEFT`, `CENTER`, `RIGHT`) and vertical proximity tiers (`CRITICAL`, `NEAR`, `MEDIUM`, `FAR`).
- Spatial anchoring maps small items (e.g. glasses, phone, keys) to detected furniture anchors (e.g. table, desk, couch, bed).

---

## 4. Technical Boundaries & Limitations

### 4.1 Perception & Optical Limitations
- **Lighting Conditions**: YOLO object detection and OCR accuracy degrade significantly under extreme low-light (< 10 lux) or strong direct backlighting.
- **Occlusions**: Objects completely covered or out of camera field of view cannot be actively perceived. NAYRA falls back to the last recorded timestamp and anchor in Object Location Memory.
- **Monocular Depth Approximation**: Depth estimates without dedicated hardware sensors (LiDAR/Stereo) are approximations based on bounding box geometry and Monocular Depth Estimation networks.

### 4.2 Voice & Acoustic Limitations
- **Background Noise**: High ambient acoustic noise (crowded streets, loud television) can degrade Whisper STT word error rates.
- **Microphone Distance**: Best performance is achieved when microphone is within 1–2 meters of the speaker.

### 4.3 Safety Disclaimers
> [!IMPORTANT]
> **Assistive Aid Disclaimer**: NAYRA is designed as an assistive companion aid and is NOT a certified medical device, autonomous navigation guide, or life-support system. It should never replace standard mobility aids such as white canes, guide dogs, or human caregivers in hazardous environments (e.g. crossing active traffic roads, subway platforms, or descending uninspected stairs).
