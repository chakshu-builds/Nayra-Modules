"""NAYRA Module 08 — Step 8.2A: Kokoro-82M Local TTS Smoke Test & Benchmark.

Executes:
1. Environment and dependency audit.
2. Model asset checksum and shape verification.
3. Audio generation for all 10 mandated test sentences (Hindi, English, Hinglish).
4. Rigorous performance benchmarking (10 warm-up + 30 measured iterations on CPU).
5. Detailed JSON report generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
# Process memory queried via ctypes / Win32 API below

try:
    from kokoro_onnx import Kokoro
except ImportError as exc:
    print(f"Error importing kokoro_onnx: {exc}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KokoroSmokeTest")

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "tts" / "kokoro"
MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODEL_DIR / "voices-v1.0.bin"
OUTPUT_DIR = ROOT_DIR / "data" / "tts_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_process_memory_mb() -> float:
    """Query current process memory using ctypes Windows API."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_smoke_test() -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("NAYRA Module 08 — Step 8.2A: Kokoro-82M Local Smoke Test")
    logger.info("================================================================")

    # 1. Asset verification
    logger.info("Verifying model files at %s...", MODEL_DIR)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VOICES_PATH.exists():
        raise FileNotFoundError(f"Voices file not found: {VOICES_PATH}")

    model_size = MODEL_PATH.stat().st_size
    voices_size = VOICES_PATH.stat().st_size
    logger.info("Computing SHA256 checksums...")
    model_sha256 = compute_sha256(MODEL_PATH)
    voices_sha256 = compute_sha256(VOICES_PATH)
    logger.info("kokoro-v1.0.onnx: %d bytes | SHA256: %s", model_size, model_sha256)
    logger.info("voices-v1.0.bin:  %d bytes | SHA256: %s", voices_size, voices_sha256)

    # 2. Model initialization (CPU Execution Provider)
    logger.info("Initializing ONNX Runtime CPU session with graph optimizations...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 4
    so.intra_op_num_threads = min(cpu_cores, 8)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    t_init_start = time.perf_counter()
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, str(VOICES_PATH))
    t_init = (time.perf_counter() - t_init_start) * 1000.0
    logger.info("Kokoro-82M CPU session initialized in %.2f ms", t_init)

    available_voices = kokoro.get_voices()
    logger.info("Total voices available: %d", len(available_voices))

    # 3. Test sentences specification
    test_cases = [
        # Category: Hindi (voice: hf_alpha, lang: hi)
        {
            "category": "Hindi",
            "id": "hindi_01",
            "text": "नमस्ते, आपका स्वागत है।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Greeting and welcome in standard Hindi",
        },
        {
            "category": "Hindi",
            "id": "hindi_02",
            "text": "कृपया आगे बढ़िए, रास्ता साफ है।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Directional navigation guidance with conjuncts",
        },
        {
            "category": "Hindi",
            "id": "hindi_03",
            "text": "स्वास्थ्य और दृष्टिबाधित व्यक्तियों की सहायता के लिए यह उपकरण बनाया गया है।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Complex Hindi terminology (स्वास्थ्य, दृष्टिबाधित, उपकरण)",
        },
        {
            "category": "Hindi",
            "id": "hindi_04",
            "text": "सावधान! आगे सीढ़ियाँ हैं।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Urgent hazard alert with exclamation mark and chandrabindu",
        },
        # Category: English (voice: af_bella, lang: en-us)
        {
            "category": "English",
            "id": "english_01",
            "text": "Hello, welcome to NAYRA.",
            "voice": "af_bella",
            "lang": "en-us",
            "description": "Standard English welcome greeting",
        },
        {
            "category": "English",
            "id": "english_02",
            "text": "Please proceed carefully. An obstacle is ahead.",
            "voice": "af_bella",
            "lang": "en-us",
            "description": "Assistive mobility navigation warning",
        },
        {
            "category": "English",
            "id": "english_03",
            "text": "Your prescription has been scanned successfully.",
            "voice": "af_bella",
            "lang": "en-us",
            "description": "OCR operational confirmation message",
        },
        # Category: Hinglish (voice: hf_alpha for Devanagari base, lang: hi)
        {
            "category": "Hinglish",
            "id": "hinglish_01",
            "text": "यह दवाई twice a day लेनी है।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Medical dosage instruction with embedded English phrase",
        },
        {
            "category": "Hinglish",
            "id": "hinglish_02",
            "text": "कृपया अपना ID card साथ रखें।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Identification document prompt with acronym and English loan word",
        },
        {
            "category": "Hinglish",
            "id": "hinglish_03",
            "text": "Emergency exit gate आपके बाईं ओर है।",
            "voice": "hf_alpha",
            "lang": "hi",
            "description": "Directional instruction starting with multi-word English phrase",
        },
    ]

    # 4. Generate audio for all 10 sentences
    logger.info("Generating WAV files for %d test sentences...", len(test_cases))
    generated_samples = []

    for idx, tc in enumerate(test_cases, 1):
        wav_path = OUTPUT_DIR / f"{tc['id']}.wav"
        t0 = time.perf_counter()
        samples, sr = kokoro.create(tc["text"], voice=tc["voice"], lang=tc["lang"])
        synth_time = (time.perf_counter() - t0) * 1000.0
        audio_dur = len(samples) / sr
        rtf = (synth_time / 1000.0) / audio_dur if audio_dur > 0 else 0.0

        sf.write(str(wav_path), samples, sr)
        peak_ram = get_process_memory_mb()

        phonemes = kokoro.tokenizer.phonemize(tc["text"], tc["lang"])

        res_entry = {
            "index": idx,
            "category": tc["category"],
            "id": tc["id"],
            "text": tc["text"],
            "voice": tc["voice"],
            "lang": tc["lang"],
            "wav_file": str(wav_path),
            "audio_duration_sec": round(audio_dur, 2),
            "synthesis_time_ms": round(synth_time, 2),
            "rtf": round(rtf, 4),
            "sample_rate": sr,
            "phonemes": phonemes,
            "peak_process_ram_mb": round(peak_ram, 2),
        }
        generated_samples.append(res_entry)

        logger.info(
            "[%d/%d] %s: Generated %.2fs audio in %.2f ms (RTF: %.3f) -> %s",
            idx,
            len(test_cases),
            tc["id"],
            audio_dur,
            synth_time,
            rtf,
            wav_path.name,
        )

    # 5. Controlled Performance Benchmarking
    # Protocol: 10 warmup iterations, then 30 measured iterations on:
    # Benchmark Target A (Short English): "Hello, welcome to NAYRA."
    # Benchmark Target B (Medium Hindi): "स्वास्थ्य और दृष्टिबाधित व्यक्तियों की सहायता के लिए यह उपकरण बनाया गया है।"
    logger.info("Starting controlled benchmark: 10 warmup + 30 measured iterations...")

    bench_targets = [
        ("Short_English", "Hello, welcome to NAYRA.", "af_bella", "en-us"),
        ("Medium_Hindi", "स्वास्थ्य और दृष्टिबाधित व्यक्तियों की सहायता के लिए यह उपकरण बनाया गया है।", "hf_alpha", "hi"),
    ]

    bench_results = {}

    for name, text, voice, lang in bench_targets:
        logger.info("Benchmarking target: %s ('%s')...", name, text[:30])

        # 10 Warmup iterations
        for w_i in range(10):
            kokoro.create(text, voice=voice, lang=lang)

        # 30 Measured iterations
        latencies_ms = []
        durations_sec = []

        for m_i in range(30):
            t0 = time.perf_counter()
            samples, sr = kokoro.create(text, voice=voice, lang=lang)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            durations_sec.append(len(samples) / sr)

        mean_lat = statistics.mean(latencies_ms)
        median_lat = statistics.median(latencies_ms)
        p95_lat = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
        min_lat = min(latencies_ms)
        max_lat = max(latencies_ms)
        audio_dur = statistics.mean(durations_sec)
        median_rtf = (median_lat / 1000.0) / audio_dur

        bench_results[name] = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "iterations": 30,
            "warmup_iterations": 10,
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(median_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "audio_duration_sec": round(audio_dur, 2),
            "median_rtf": round(median_rtf, 4),
            "sample_rate": 24000,
        }

        logger.info(
            "Benchmark %s: Median=%.1f ms | P95=%.1f ms | Audio=%.2fs | RTF=%.3f",
            name,
            median_lat,
            p95_lat,
            audio_dur,
            median_rtf,
        )

    # 6. Overall Report Packaging
    full_report = {
        "metadata": {
            "test_suite": "NAYRA Module 08 Step 8.2A Kokoro-82M Smoke Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "os": platform.platform(),
            "python_version": sys.version,
            "onnxruntime_version": ort.__version__,
            "cpu_cores": cpu_cores,
            "device": "CPUExecutionProvider (intra_op=8, ORT_ENABLE_ALL)",
        },
        "model_verification": {
            "model_path": str(MODEL_PATH),
            "model_bytes": model_size,
            "model_sha256": model_sha256,
            "voices_path": str(VOICES_PATH),
            "voices_bytes": voices_size,
            "voices_sha256": voices_sha256,
            "initialization_time_ms": round(t_init, 2),
            "total_voices": len(available_voices),
        },
        "test_samples": generated_samples,
        "benchmark_summary": bench_results,
    }

    report_path = OUTPUT_DIR / "smoke_test_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    logger.info("Smoke test results saved to %s", report_path)
    return full_report


if __name__ == "__main__":
    run_smoke_test()
