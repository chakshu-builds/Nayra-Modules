# NAYRA Module 09 — Phone Microphone Input & Voice Command Recognition

Module 09 provides real-time microphone audio streaming from the Android phone (via IP Webcam's `/audio.wav` endpoint), fast 100% offline speech recognition for voice commands in English, Hindi, and Hinglish, and coordinates **Phase 3: End-to-End Voice-Controlled Read Aloud**.

---

## 1. Directory Structure

```
09_voice/
    __init__.py                  # Public package exports (including ReadAloudPipeline)
    command_matcher.py           # CommandIntent (READ, STOP) & regex/phrase matcher
    audio_stream_receiver.py     # Live HTTP audio streamer from IP Webcam (/audio.wav) + VAD
    voice_command_recognizer.py  # Offline Whisper-tiny INT8 speech-to-intent engine
    read_aloud_pipeline.py       # Phase 3 coordinator: Voice STT + Camera Frame + OCR + TTS
    run_read_aloud.py            # Phase 3 standalone CLI runner (live network or simulation)
    verify_phone_audio.py        # Interactive diagnostic and live audio verification tool
    tests/
        __init__.py              # Test suite package
        conftest.py              # Pytest setup and path imports
        test_command_matcher.py  # Unit tests for intent classification
        test_audio_receiver.py   # Unit tests for stream normalization and VAD
        test_recognizer.py       # Integration tests with synthesized voice commands
        test_read_aloud_pipeline.py # Unit & full real-model E2E integration test suite
    README.md                    # Architecture and usage documentation
```

Root entrypoints:
- `voice.py`: Standard Python identifier alias (`from voice import PhoneAudioReceiver, VoiceCommandRecognizer, CommandIntent, ReadAloudPipeline`).
- `ocr.py`: Standard Python alias (`from ocr import read_text, OCRResult`).
- `tts.py`: Standard Python alias (`from tts import TTSManager, PriorityLevel`).

---

## 2. Supported Voice Commands & Intent Mapping

| Spoken Voice Command | Language Mode | Mapped Intent | Triggered System Action |
| :--- | :--- | :---: | :--- |
| **"NAYRA, read the text"** | English | `CommandIntent.READ` | Acquire frame -> Run Module 07 OCR -> Speak text with Module 08 TTS |
| **"Read this"** | English | `CommandIntent.READ` | Acquire frame -> Run Module 07 OCR -> Speak text with Module 08 TTS |
| **"Mere liye padho" / "मेरे लिए पढ़ो"** | Hindi | `CommandIntent.READ` | Acquire frame -> Run Module 07 OCR -> Speak text with Module 08 TTS |
| **"Stop reading"** | English | `CommandIntent.STOP` | Immediately halt active Module 08 TTS playback (< 1 ms latency) |
| **"Text reading band karo"** | Hinglish | `CommandIntent.STOP` | Immediately halt active Module 08 TTS playback (< 1 ms latency) |

---

## 3. End-to-End Architecture (Phase 3)

```
┌────────────────────────────────────────────────────────┐
│             Android Phone (IP Webcam App)              │
│       • Microphone permission granted                  │
│       • Camera stream: http://<ip>:8080/video          │
│       • Audio stream : http://<ip>:8080/audio.wav      │
└──────────┬──────────────────────────────────┬──────────┘
           │ HTTP GET /audio.wav              │ HTTP GET /video
           ▼                                  ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  Module 09: Audio Receiver   │    │  Module 01/02: Camera Stream │
│  ├── Dynamic WAV Parser      │    │  └── FrameReader Buffer      │
│  └── Energy-based VAD        │    └──────────────┬───────────────┘
└──────────┬───────────────────┘                   │
           │ 16 kHz PCM chunk                      │ On READ trigger
           ▼                                       │ (debounced >= 2.0s)
┌──────────────────────────────┐                   │
│  Module 09: Whisper-tiny INT8│                   │
│  └── Intent: READ or STOP    │                   │
└──────────┬───────────────────┘                   │
           │                                       │
     ┌─────┴──────────────────┐                    │
     │                        │                    │
[STOP Intent]            [READ Intent]             │
     │                        │                    ▼
     │                        │         ┌──────────────────────────────┐
     │                        └────────►│  Module 07: Unified OCR      │
     │                                  │  ├── DBNet Line Detection    │
     │                                  │  └── PP-OCRv5 Text Recog     │
     │                                  └──────────────┬───────────────┘
     │                                                 │ OCRResult.combined_text
     ▼                                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                  Module 08: TTS Manager                      │
│  ├── Immediate stop() if STOP intent                         │
│  ├── Conservative normalization & sentence chunking          │
│  ├── Kokoro-82M neural speech synthesis (English + Hindi)    │
│  └── Non-blocking PortAudio playback on laptop speakers      │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Key Guarantees & Safeguards

1. **Decoupled Asynchronous Concurrency**:
   Microphone streaming and STT run on background worker threads, completely independent of camera frame acquisition and OCR inference. Camera FPS never stutters while listening for voice commands.
2. **2.0s Debounce State Gate**:
   A single vocal utterance ("Read this") cannot trigger repeated OCR passes or overlapping audio responses due to acoustic room reverberation.
3. **Graceful Verbal Fallbacks**:
   - Camera frame unavailable: *"Camera frame not available."* / *"कैमरा उपलब्ध नहीं है।"*
   - Scene contains no printed text: *"No text detected."* / *"कोई लिखावट नहीं मिली।"*
4. **Instant Stop Interruption**:
   Speaking *"Stop reading"* or *"Text reading band karo"* halts active speaker output immediately and cancels remaining queued text chunks.
5. **100% Offline Local Execution**:
   All three subsystems (Whisper STT, DBNet + PP-OCRv5, Kokoro TTS) operate entirely offline without network latency, cloud costs, or external API dependencies.

---

## 5. How to Run Phase 3

### Option A: Standalone Live Network Run (with Android Phone)
1. Open **IP Webcam** on your Android phone, enable **Audio connection**, and tap **Start server**.
2. Run:
```bash
python 09_voice/run_read_aloud.py --ip 192.168.0.101
```
3. Speak *"NAYRA, read the text"* or *"मेरे लिए पढ़ो"*.
4. Speak *"Stop reading"* or *"Text reading band karo"* to interrupt.

### Option B: Standalone Simulation Run (Offline / No Phone Needed)
```bash
python 09_voice/run_read_aloud.py --simulate
```
Controls in console:
- Press `[R] + Enter` to simulate reading the test image.
- Press `[S] + Enter` to simulate stopping speech.
- Press `[Q] + Enter` to exit.

### Option C: Run Automated Test Suite
```bash
# Fast unit & mock tests:
python -m pytest 09_voice/tests/test_audio_receiver.py 09_voice/tests/test_command_matcher.py 09_voice/tests/test_read_aloud_pipeline.py -k "not test_real_models_end_to_end"

# Complete end-to-end test including real Whisper + real OCR + real Kokoro:
python -m pytest 09_voice/tests/test_read_aloud_pipeline.py -v
```
