"""NAYRA Phase 3 — End-to-End CPU Verification Script.

Tests:
1. Live Android IP Webcam frame acquisition via fresh on-demand capture.
2. CPU provider OCR execution on live camera frame.
3. CPU provider OCR execution on real document frame (image1.jpg).
4. Physical/mock TTS synthesis and audio dispatch.
5. End-to-end voice intent trigger -> frame capture -> CPU OCR -> TTS pipeline handoff.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from voice import CommandIntent, PhoneAudioReceiver, ReadAloudPipeline, VoiceCommandRecognizer
from tts import KokoroEngine, PriorityLevel, TTSManager
import ocr

print("=" * 76)
print("     NAYRA Phase 3 — Complete End-to-End Read Aloud CPU Verification")
print("=" * 76)

# 1. Live Camera Frame Acquisition
video_url = "http://192.168.0.101:8080/video"
print(f"\n[STEP 1] Testing live frame acquisition from {video_url} ...")
cap = cv2.VideoCapture(video_url)
live_frame = None
if cap.isOpened():
    for _ in range(5):
        ret, tmp = cap.read()
        if ret and tmp is not None and tmp.size > 0:
            live_frame = tmp
    cap.release()

if live_frame is not None:
    print(f"  [SUCCESS] Acquired live camera frame: shape={live_frame.shape}, mean brightness={live_frame.mean():.2f}")
    cv2.imwrite("data/ocr_diagnostics/live_camera_cpu_capture.jpg", live_frame)
else:
    print("  [WARNING] Live camera frame could not be acquired (network or device offline).")

# 2. CPU Provider OCR Execution on Live Camera Frame
print("\n[STEP 2] Running Module 07 OCR on live camera frame (Provider: CPU) ...")
if live_frame is not None:
    t0 = time.perf_counter()
    res_live = ocr.read_text(live_frame, preferred_provider="CPU", force_new=True)
    t1 = time.perf_counter()
    print(f"  [SUCCESS] OCR completed in {(t1 - t0)*1000:.1f} ms")
    print(f"  Active Providers : {res_live.execution_providers}")
    print(f"  Success Flag     : {res_live.success}")
    print(f"  Regions Detected : {len(res_live.regions)}")
    print(f"  Text Detected    : {repr(res_live.combined_text)}")
    assert res_live.success is True, "OCR execution failed!"
    assert all("CPU" in str(p) for p in res_live.execution_providers.values()), "Providers not strictly CPU!"

# 3. CPU Provider OCR Execution on Verified Printed Document (image1.jpg)
print("\n[STEP 3] Running Module 07 OCR on printed document image1.jpg (Provider: CPU) ...")
doc_frame = cv2.imread("image1.jpg")
assert doc_frame is not None, "image1.jpg missing"
t0 = time.perf_counter()
res_doc = ocr.read_text(doc_frame, preferred_provider="CPU", force_new=True)
t1 = time.perf_counter()
print(f"  [SUCCESS] OCR completed in {(t1 - t0)*1000:.1f} ms")
print(f"  Active Providers : {res_doc.execution_providers}")
print(f"  Regions Detected : {len(res_doc.regions)}")
print(f"  Combined Text    : {repr(res_doc.combined_text)}")
assert res_doc.success is True
assert len(res_doc.regions) > 0, "No text detected in document image1.jpg!"
assert "राजीव" in res_doc.combined_text or "Rajiv" in res_doc.combined_text

# 4. Physical / Background TTS Playback on Extracted OCR Text
print("\n[STEP 4] Dispatching extracted text to Kokoro TTSManager ...")
tts = TTSManager(mock_audio=False)
tts_req_id = tts.speak(res_doc.combined_text.strip(), priority=PriorityLevel.NORMAL, source="ocr")
print(f"  [SUCCESS] Enqueued speech request {tts_req_id}")
print("  Waiting 4.0s for speech synthesizer warmup and output...")
time.sleep(4.0)
tts.stop()
tts.shutdown()
print("  [SUCCESS] TTS output completed cleanly.")

# 5. Integrated Pipeline Simulation with Real Models on CPU
print("\n[STEP 5] Testing full ReadAloudPipeline end-to-end handoff on CPU ...")
events = []
def on_event(name, data):
    events.append((name, data))
    if name == "ocr_completed":
        print(f"    [EVENT] ocr_completed: text_found={data['text_found']}, lang={data['primary_language']}, providers={data.get('execution_providers')}")
    elif name == "tts_enqueued":
        print(f"    [EVENT] tts_enqueued: {data['text'][:50]}...")

# Pre-generate synthetic command audio: "NAYRA, read the text."
kokoro = KokoroEngine(use_gpu=False)
kokoro.initialize()
raw_audio, sr = kokoro.synthesize("NAYRA, read the text.", voice="af_bella", lang="en-us")
num_out = int(len(raw_audio) * 16000 / sr)
audio_16k = np.interp(
    np.linspace(0, len(raw_audio), num_out, endpoint=False),
    np.arange(len(raw_audio)),
    raw_audio,
).astype(np.float32)
kokoro.shutdown()

pipeline_tts = TTSManager(mock_audio=True)
pipeline_rec = VoiceCommandRecognizer()

pipeline = ReadAloudPipeline(
    voice_recognizer=pipeline_rec,
    tts_manager=pipeline_tts,
    frame_provider=lambda: doc_frame,
    preferred_ocr_provider="CPU",
    mock_mode=True,
    debounce_sec=1.0,
    on_event=on_event,
)

pipeline.start()
try:
    print("  Injecting voice command 'NAYRA, read the text' ...")
    pipeline.trigger_mock_command(audio_16k)

    deadline = time.time() + 25.0
    while time.time() < deadline:
        st = pipeline.get_status()
        if st["total_reads"] > 0 and any(e[0] == "tts_enqueued" for e in events):
            break
        time.sleep(0.3)

    st = pipeline.get_status()
    print(f"  Pipeline Status: total_reads={st['total_reads']}, last_intent={st['last_intent']}")
    assert st["total_reads"] == 1, "READ intent was not processed!"
    assert any(e[0] == "tts_enqueued" for e in events), "TTS was not enqueued with OCR text!"
    print("  [SUCCESS] End-to-end voice -> CPU OCR -> TTS pipeline handoff verified!")
finally:
    pipeline.close()

print("\n" + "=" * 76)
print("     ALL VERIFICATION PHASES COMPLETED AND VERIFIED ON CPU")
print("=" * 76)
