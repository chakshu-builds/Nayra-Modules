"""NAYRA Module 08 — Comprehensive Performance & Coexistence Benchmark.

Measures:
1. Cold engine initialization time
2. Warm synthesis latency & RTF (Hindi, English, Hinglish)
3. Time-to-First-Audio (TTFA)
4. Long paragraph streaming playback
5. Stop & emergency preemption latency
6. Camera feed loop FPS before and during active TTS synthesis
7. Peak RAM and VRAM utilization
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tts
from tts import AudioPlayer, KokoroEngine, PriorityLevel, TTSManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-TTSBenchmark")


# Windows memory query structures
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
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
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_ram_mb() -> float:
    try:
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        success = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        if success:
            return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def get_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def benchmark_tts():
    logger.info("================================================================")
    logger.info("NAYRA Module 08 — Step 8.2B TTS Performance & Coexistence Benchmark")
    logger.info("================================================================")

    report: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "os": "Windows 11",
            "cpu_cores": os.cpu_count() or 12,
            "device": "CPUExecutionProvider (intra_op=8, ORT_ENABLE_ALL)",
        }
    }

    baseline_ram = get_process_ram_mb()
    logger.info("Baseline process RAM: %.2f MB", baseline_ram)

    # 1. Cold Engine Initialization
    logger.info("1. Measuring cold engine initialization time...")
    t0 = time.perf_counter()
    engine = KokoroEngine(use_gpu=False)
    engine.initialize()
    cold_init_ms = (time.perf_counter() - t0) * 1000.0
    post_init_ram = get_process_ram_mb()
    logger.info("KokoroEngine initialized in %.2f ms (RAM: %.2f MB).", cold_init_ms, post_init_ram)

    report["initialization"] = {
        "cold_init_ms": round(cold_init_ms, 2),
        "baseline_ram_mb": round(baseline_ram, 2),
        "post_init_ram_mb": round(post_init_ram, 2),
    }

    # Initialize Player & TTSManager
    player = AudioPlayer(mock_mode=False)
    manager = TTSManager(engine=engine, player=player, mock_audio=False)

    # 2. Warm Synthesis & RTF Benchmarks
    logger.info("2. Measuring warm synthesis latency and RTF across languages...")
    targets = [
        ("English", "Please proceed carefully. An obstacle is ahead.", "af_bella", "en-us"),
        ("Hindi", "नमस्ते, आपका स्वागत है। कृपया आगे बढ़िए, रास्ता साफ है।", "hf_alpha", "hi"),
        ("Hinglish", "यह दवाई twice a day लेनी है। कृपया अपना ID card साथ रखें।", "hf_alpha", "hi"),
    ]

    synthesis_bench = {}
    for name, text, voice, lang in targets:
        logger.info("Benchmarking %s...", name)
        t_synth = time.perf_counter()
        samples, sr = engine.synthesize(text, voice=voice, lang=lang)
        synth_ms = (time.perf_counter() - t_synth) * 1000.0
        audio_dur = len(samples) / sr
        rtf = (synth_ms / 1000.0) / audio_dur if audio_dur > 0 else 0.0

        synthesis_bench[name] = {
            "text": text,
            "audio_duration_sec": round(audio_dur, 2),
            "synthesis_time_ms": round(synth_ms, 2),
            "rtf": round(rtf, 4),
        }
        logger.info("%s: Dur=%.2fs | Synth=%.2fms | RTF=%.3f", name, audio_dur, synth_ms, rtf)

    report["synthesis_benchmarks"] = synthesis_bench

    # 3. Time to First Audio (TTFA) on Live Playback
    logger.info("3. Measuring Time to First Audio (TTFA) on live hardware playback...")
    ttfa_tests = [
        ("English_Alert", "Navigation guidance updated.", PriorityLevel.PERCEPTION, "en-us"),
        ("Hindi_Alert", "सावधान! आगे सीढ़ियाँ हैं।", PriorityLevel.CRITICAL, "hi"),
    ]
    ttfa_results = {}
    for name, text, prio, lang in ttfa_tests:
        t_start = time.perf_counter()
        req_id = manager.speak(text, priority=prio, language_hint=lang)
        ttfa = None
        while time.perf_counter() - t_start < 8.0:
            if player.is_playing():
                ttfa = (time.perf_counter() - t_start) * 1000.0
                break
            time.sleep(0.01)

        while manager.is_speaking():
            time.sleep(0.05)
        total_time = (time.perf_counter() - t_start) * 1000.0

        ttfa_results[name] = {
            "text": text,
            "priority": prio,
            "ttfa_ms": round(ttfa, 2) if ttfa else None,
            "total_playback_ms": round(total_time, 2),
        }
        logger.info("%s: TTFA=%.2f ms | Total=%.2f ms", name, ttfa if ttfa else -1, total_time)

    report["ttfa_benchmarks"] = ttfa_results

    # 4. Long Multi-Sentence Paragraph Pipelined Streaming
    logger.info("4. Testing long paragraph pipelined chunk streaming...")
    long_paragraph = (
        "Dr. Sharma prescribed Paracetamol 500mg. "
        "Take one tablet twice a day after meals. "
        "यह दवाई नियमित समय पर लेनी है। "
        "सावधान! यदि कोई समस्या हो तो तुरंत चिकित्सक से संपर्क करें।"
    )
    t_para_start = time.perf_counter()
    req_id = manager.speak(long_paragraph, priority=PriorityLevel.NORMAL)

    para_ttfa = None
    while time.perf_counter() - t_para_start < 10.0:
        if player.is_playing():
            para_ttfa = (time.perf_counter() - t_para_start) * 1000.0
            break
        time.sleep(0.01)

    while manager.is_speaking():
        time.sleep(0.05)
    para_total = (time.perf_counter() - t_para_start) * 1000.0

    report["long_paragraph_streaming"] = {
        "text_length_chars": len(long_paragraph),
        "ttfa_ms": round(para_ttfa, 2) if para_ttfa else None,
        "total_duration_ms": round(para_total, 2),
    }
    logger.info("Long paragraph: TTFA=%.2f ms (Chunk 0 start) | Total Duration=%.2f ms", para_ttfa if para_ttfa else -1, para_total)

    # 5. Stop and Emergency Preemption Latency
    logger.info("5. Measuring stop & emergency preemption latency...")
    # Start long reading
    manager.speak("This is a long continuous reading that will be interrupted by an emergency.", priority=PriorityLevel.NORMAL)
    time.sleep(0.3)  # Allow synthesis/playback to start

    t_stop_call = time.perf_counter()
    manager.stop()
    stop_latency_ms = (time.perf_counter() - t_stop_call) * 1000.0
    logger.info("manager.stop() completed in %.3f ms (is_playing: %s).", stop_latency_ms, player.is_playing())

    # Preemption latency: Start normal speech, then send Critical alert
    manager.speak("Continuing normal OCR reading from camera capture.", priority=PriorityLevel.NORMAL)
    time.sleep(0.2)

    t_alert_call = time.perf_counter()
    alert_id = manager.speak("रुको! खतरा है।", priority=PriorityLevel.CRITICAL, interrupt=True, language_hint="hi")
    alert_ttfa = None
    while time.perf_counter() - t_alert_call < 5.0:
        if player.is_playing():
            alert_ttfa = (time.perf_counter() - t_alert_call) * 1000.0
            break
        time.sleep(0.01)

    while manager.is_speaking():
        time.sleep(0.05)

    report["preemption_and_stop"] = {
        "stop_latency_ms": round(stop_latency_ms, 3),
        "critical_preemption_ttfa_ms": round(alert_ttfa, 2) if alert_ttfa else None,
    }
    logger.info("Critical preemption TTFA: %.2f ms.", alert_ttfa if alert_ttfa else -1)

    # 6. Camera Loop Coexistence Benchmark
    logger.info("6. Measuring camera processing loop FPS before and during active TTS...")
    # Simulate a 1080p camera loop with frame copying and dummy preprocessing
    frame_h, frame_w = 1080, 1920
    dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Phase A: Baseline camera loop (50 iterations)
    t_cam_start = time.perf_counter()
    for _ in range(50):
        # Frame copy + blur + grayscale (standard perception pre-processing)
        f_copy = dummy_frame.copy()
        gray = cv2.cvtColor(f_copy, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(gray, (5, 5), 0)
    baseline_cam_sec = time.perf_counter() - t_cam_start
    baseline_fps = 50 / baseline_cam_sec

    # Phase B: Camera loop during active background TTS synthesis
    manager.speak("Testing camera frame rate stability while synthesizing speech in background.", priority=PriorityLevel.NORMAL)
    t_cam_tts = time.perf_counter()
    for _ in range(50):
        f_copy = dummy_frame.copy()
        gray = cv2.cvtColor(f_copy, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(gray, (5, 5), 0)
    tts_cam_sec = time.perf_counter() - t_cam_tts
    tts_fps = 50 / tts_cam_sec

    while manager.is_speaking():
        time.sleep(0.05)

    fps_drop_percent = ((baseline_fps - tts_fps) / baseline_fps) * 100.0
    report["camera_coexistence"] = {
        "baseline_camera_fps": round(baseline_fps, 2),
        "during_tts_camera_fps": round(tts_fps, 2),
        "fps_drop_percent": round(fps_drop_percent, 2),
    }
    logger.info("Camera FPS: Baseline=%.2f FPS | During TTS=%.2f FPS (Drop: %.2f%%)", baseline_fps, tts_fps, fps_drop_percent)

    # 7. Final Resource Footprint
    peak_ram = get_process_ram_mb()
    peak_vram = get_vram_mb()
    report["resource_footprint"] = {
        "baseline_ram_mb": round(baseline_ram, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "peak_vram_mb": round(peak_vram, 2),
    }
    logger.info("Peak RAM: %.2f MB | Peak VRAM: %.2f MB", peak_ram, peak_vram)

    # Shutdown
    manager.shutdown()

    # Save report
    out_path = ROOT_DIR / "data" / "tts_samples" / "benchmark_8_2b_report.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    logger.info("Benchmark complete! Saved to %s", out_path)
    return report


if __name__ == "__main__":
    benchmark_tts()
