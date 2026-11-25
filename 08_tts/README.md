# NAYRA Module 08 — Offline Multilingual Text-to-Speech (TTS) Subsystem

Module 08 provides fully offline, low-latency, natural Hindi + English speech synthesis and asynchronous audio playback for NAYRA assistive glasses.

Powered by the lightweight **Kokoro-82M ONNX** neural model (StyleTTS 2 / ISTFTNet) and PortAudio (`sounddevice`), Module 08 executes locally without internet access or paid APIs.

---

## 1. Directory Structure

```
08_tts/
    __init__.py              # Public package exports
    tts_manager.py           # Central TTS coordinator & background worker
    request_schema.py        # Typed SpeechRequest and PriorityLevel definitions
    normalizer.py            # Conservative text normalizer & script router
    chunker.py               # Sentence & clause chunker for pipelined streaming
    benchmark_tts.py         # Automated performance & coexistence benchmark suite
    engines/
        __init__.py          # Engines package exports
        base.py              # Abstract BaseTTSEngine interface & exceptions
        kokoro_engine.py     # Kokoro-82M ONNX synthesis engine (CPU & CUDA)
    audio/
        __init__.py          # Audio package exports
        audio_player.py      # Non-blocking PortAudio stream player with mock fallback
        audio_queue.py       # Thread-safe bounded priority queue with preemption
    tests/
        __init__.py          # Test suite package
        conftest.py          # Pytest fixtures and sys.path configuration
        test_normalizer.py   # Unit tests for text normalization & script detection
        test_chunker.py      # Unit tests for danda and abbreviation chunking
        test_queue.py        # Unit tests for priority ordering & preemption purge
        test_engine.py       # Integration tests for Kokoro ONNX model synthesis
        test_tts_manager.py  # Lifecycle, pause/resume, and stop integration tests
        test_physical_playback.py # Physical speaker/headphone live audio test
    README.md                # Subsystem architecture & integration documentation
```

Root alias:
- `tts.py`: Exposes `from tts import TTSManager, SpeechRequest, PriorityLevel` to eliminate numeric directory import limitations.

---

## 2. Public API Reference

### 2.1 Initialization
```python
from tts import TTSManager

# Standard CPU execution (default: 8 intra-op threads, ORT_ENABLE_ALL)
tts = TTSManager()
```

### 2.2 Speech Synthesis & Playback (`speak`)
```python
# Normal reading (Priority 2, non-interrupting, non-blocking)
req_id = tts.speak("Paracetamol 500mg. Take one tablet twice a day.")

# Critical hazard alert (Priority 0, interrupts active playback immediately)
alert_id = tts.speak("सावधान! आगे बाधा है।", priority=0, interrupt=True)
```

**Parameters for `speak()`:**
- `text (str)`: Input text string in Devanagari Hindi, English Latin, or Hinglish.
- `priority (int)`: Priority level:
  - `PriorityLevel.CRITICAL (0)`: Safety alerts (obstacles, stairs, drop-offs).
  - `PriorityLevel.PERCEPTION (1)`: Important perception updates (faces, objects).
  - `PriorityLevel.NORMAL (2)`: Normal OCR reading, narration.
- `source (str)`: Origin component (`"manual"`, `"ocr"`, `"safety"`, `"perception"`).
- `interrupt (bool)`: If `True`, halts lower-priority active playback immediately.
- `language_hint (Optional[str])`: Manual override (`"hi"`, `"en-us"`, `"auto"`).

### 2.3 Playback Controls
- `tts.stop()`: Immediately stops physical audio playback within < 1 ms, clears current buffers, and marks the active request as `CANCELLED`.
- `tts.pause()`: Pauses active playback stream without losing current buffer position.
- `tts.resume()`: Resumes playback from paused position.
- `tts.is_speaking() -> bool`: Returns `True` if actively synthesizing or playing audio.
- `tts.get_status() -> dict`: Returns structured engine status, queue size, and active request.
- `tts.shutdown(timeout=3.0)`: Gracefully stops workers, unloads model, and closes audio streams.

### 2.4 Pause/Resume Semantics
- **During Active Playback:** `AudioPlayer` holds the PortAudio stream buffer in place. The background worker pauses before dispatching subsequent chunks.
- **During Active Synthesis:** The in-flight ONNX inference call for the current chunk finishes, but playback is withheld until `resume()` is called.

---

## 3. Pipelined Chunking & Low Latency (TTFA)

Kokoro-82M operates as an end-to-end neural synthesizer. For long OCR paragraphs, synthesizing the entire text before playback would cause a 5–10 second delay.

`TTSManager` implements **Pipelined Streaming**:
1. `TextChunker` breaks text into natural sentences (delimited by Hindi danda `।`, `!`, `?`, `.`, or clause boundaries for long sentences).
2. **Chunk 0** is synthesized and immediately sent to `AudioPlayer.play()`.
3. Physical playback of Chunk 0 begins (**TTFA: ~650–900 ms**).
4. Subsequent chunks (Chunk 1, Chunk 2, ...) are synthesized in the background concurrently while the user listens to Chunk 0.

---

## 4. OCR Integration Boundary (Module 07 → Module 08)

To preserve architectural modularity, **never pass complete `OCRResult` objects into `TTSManager.speak()`**. Use an adapter callback that extracts the validated `combined_text`:

```python
from tts import TTSManager, PriorityLevel
from ocr import OCRResult

def ocr_speech_callback(ocr_result: OCRResult, tts_manager: TTSManager) -> None:
    """Safely convert completed OCRResult into speech output."""
    if not ocr_result or not ocr_result.success:
        return

    text = ocr_result.combined_text.strip()
    if not text:
        return

    # Language hint derived from OCR primary language classification
    lang_hint = "hi" if ocr_result.primary_language.value == "HINDI" else "en-us"

    # Enqueue as normal background reading (Priority 2)
    tts_manager.speak(
        text=text,
        priority=PriorityLevel.NORMAL,
        source="ocr",
        interrupt=False,
        language_hint=lang_hint,
    )
```

---

## 5. Verification & Testing

### Running Unit & Integration Tests:
```bash
python -m pytest 08_tts/tests/test_normalizer.py 08_tts/tests/test_chunker.py 08_tts/tests/test_queue.py 08_tts/tests/test_tts_manager.py 08_tts/tests/test_engine.py
```

### Running Physical Speaker Playback Test:
```bash
python 08_tts/tests/test_physical_playback.py
```

### Running Performance & Coexistence Benchmark:
```bash
python 08_tts/benchmark_tts.py
```
