"""Performance and Resource Efficiency Profiling Script for NAYRA MVP.

Measures:
1. Pure orchestrator and service construction latency.
2. Cold model loading latency (YOLOv10n ONNX, Kokoro TTS, Faster-Whisper).
3. Query dispatch latency (mean, median, 95th percentile, max) over 100 mixed queries.
4. Memory footprint: Python heap (tracemalloc) vs. native OS Process Working Set (RSS).
5. Active thread leak detection before and after sustained workloads.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import json
import os
import sys
import threading
import time
import tracemalloc

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistant_orchestrator import AssistantOrchestrator
from daily_assistance import DailyService
from obstacle_awareness import ObstacleService
from scene_understanding import SceneService
from alzheimer_assistance import AlzheimerService
from safety_sos import SafetyService
from voice_interaction import VoiceManager


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
    ]


def get_native_process_rss_mb() -> float:
    """Return actual Windows Process Working Set (RSS) in Megabytes."""
    try:
        k32 = ctypes.windll.kernel32
        hProc = k32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = k32.K32GetProcessMemoryInfo(hProc, ctypes.byref(c), c.cb)
        if ok:
            return round(c.WorkingSetSize / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def run_profiling():
    print("=== NAYRA SOFTWARE PERFORMANCE & RESOURCE PROFILING (PHASE 3) ===")
    tracemalloc.start()
    gc.collect()

    # Step 1: Thread count baseline and initial RSS
    baseline_threads = threading.active_count()
    baseline_rss_mb = get_native_process_rss_mb()
    print(f"[Baseline] Active threads: {baseline_threads} | Baseline Process RSS: {baseline_rss_mb} MB")

    # Step 2: Pure Orchestration Construction Latency (without heavy model weights)
    t0 = time.perf_counter()
    orchestrator = AssistantOrchestrator()
    daily_mod = DailyService()
    obstacle_mod = ObstacleService()
    scene_mod = SceneService()
    alzheimer_mod = AlzheimerService()
    sos_mod = SafetyService()
    voice_mgr = VoiceManager(tts_backend=lambda text, lang, prio: None)

    orchestrator.register_module("14_daily_assistance", daily_mod)
    orchestrator.register_module("11_obstacle_awareness", obstacle_mod)
    orchestrator.register_module("12_scene_understanding", scene_mod)
    orchestrator.register_module("13_alzheimer_assistance", alzheimer_mod)
    orchestrator.register_module("15_safety_sos", sos_mod)
    orchestrator.register_module("16_voice_interaction", voice_mgr)

    init_ok = orchestrator.initialize_all()
    t_construction_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Orchestration Construction] Modules initialized: {init_ok} in {t_construction_ms:.2f} ms")

    # Step 3: Realistic Cold-Start Model Weight Loading Benchmark
    print("\n[Cold Model Loading Benchmark]:")
    model_load_metrics = {}

    # 3a: YOLOv10n ONNX
    yolo_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov10n.onnx")
    if os.path.exists(yolo_path):
        import onnxruntime as ort
        t_yolo_start = time.perf_counter()
        _ = ort.InferenceSession(yolo_path, providers=["CPUExecutionProvider"])
        t_yolo_ms = (time.perf_counter() - t_yolo_start) * 1000.0
        model_load_metrics["yolov10n_onnx_load_ms"] = round(t_yolo_ms, 2)
        print(f"  - YOLOv10n ONNX Session Load: {t_yolo_ms:.2f} ms")

    # 3b: Kokoro TTS ONNX
    kokoro_model = os.path.join(os.path.dirname(__file__), "..", "models", "tts", "kokoro", "kokoro-v1.0.onnx")
    kokoro_voices = os.path.join(os.path.dirname(__file__), "..", "models", "tts", "kokoro", "voices-v1.0.bin")
    if os.path.exists(kokoro_model) and os.path.exists(kokoro_voices):
        from kokoro_onnx import Kokoro
        t_kokoro_start = time.perf_counter()
        _ = Kokoro(kokoro_model, kokoro_voices)
        t_kokoro_ms = (time.perf_counter() - t_kokoro_start) * 1000.0
        model_load_metrics["kokoro_tts_onnx_load_ms"] = round(t_kokoro_ms, 2)
        print(f"  - Kokoro-82M TTS ONNX Load:   {t_kokoro_ms:.2f} ms")

    # 3c: Faster-Whisper tiny STT
    whisper_dir = os.path.join(os.path.dirname(__file__), "..", "models", "stt", "whisper")
    if os.path.exists(whisper_dir):
        from faster_whisper import WhisperModel
        t_whisper_start = time.perf_counter()
        _ = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=whisper_dir)
        t_whisper_ms = (time.perf_counter() - t_whisper_start) * 1000.0
        model_load_metrics["whisper_stt_tiny_load_ms"] = round(t_whisper_ms, 2)
        print(f"  - Faster-Whisper Tiny Load:   {t_whisper_ms:.2f} ms")

    total_model_load_ms = sum(model_load_metrics.values())
    full_cold_start_ms = round(t_construction_ms + total_model_load_ms, 2)
    print(f"  -> Total Cold Model Weights Load: {total_model_load_ms:.2f} ms")
    print(f"  -> Full Cold Startup (Construction + Models): {full_cold_start_ms:.2f} ms")

    # Step 4: Query Dispatch Latency Benchmark (100 queries)
    test_queries = [
        "Mera phone kahan hai?",
        "Aage rasta saaf hai kya?",
        "Saamne ka scene batao.",
        "Saamne kaun khada hai?",
        "Shaam ko 6 baje walk par jaane ka reminder lagao.",
        "Mera chashma kahan hai?",
        "Mere sabhi reminders batao.",
        "Madad chahiye, bachao!",
        "Galti se bol diya, cancel karo.",
        "Mera laptop kahan rakha hai?",
    ]

    latencies_ms = []
    current_mem, peak_mem_start = tracemalloc.get_traced_memory()

    print(f"\n[Benchmarking] Running 100 simulated multilingual voice queries...")
    for i in range(100):
        q = test_queries[i % len(test_queries)]
        q_start = time.perf_counter()
        resp = orchestrator.process_voice_query(q)
        q_elapsed = (time.perf_counter() - q_start) * 1000.0
        latencies_ms.append(q_elapsed)

    latencies_sorted = sorted(latencies_ms)
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    median_lat = latencies_sorted[len(latencies_sorted) // 2]
    p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99_lat = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    max_lat = latencies_sorted[-1]
    min_lat = latencies_sorted[0]

    print(f"[Latency Results across 100 queries]:")
    print(f"  - Min:    {min_lat:.3f} ms")
    print(f"  - Mean:   {mean_lat:.3f} ms")
    print(f"  - Median: {median_lat:.3f} ms")
    print(f"  - P95:    {p95_lat:.3f} ms")
    print(f"  - P99:    {p99_lat:.3f} ms")
    print(f"  - Max:    {max_lat:.3f} ms")

    # Step 5: Memory footprint and leak check
    gc.collect()
    current_mem_end, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_growth_kb = (current_mem_end - current_mem) / 1024.0
    peak_python_heap_mb = peak_mem / (1024.0 * 1024.0)
    final_process_rss_mb = get_native_process_rss_mb()

    print(f"\n[Memory Profiling - Distinguishing Python Heap vs OS Working Set]:")
    print(f"  - Python Tracked Heap (Peak): {peak_python_heap_mb:.2f} MB")
    print(f"  - Python Heap Growth (100 queries): {mem_growth_kb:.2f} KB")
    print(f"  - Total OS Process Working Set (RSS with models loaded): {final_process_rss_mb:.2f} MB")

    # Step 6: Shutdown and Thread Leak Audit
    orchestrator.shutdown_all()
    time.sleep(0.1)
    final_threads = threading.active_count()
    print(f"\n[Thread Leak Audit]:")
    print(f"  - Threads at start: {baseline_threads}")
    print(f"  - Threads after shutdown: {final_threads}")
    thread_leak = final_threads - baseline_threads
    print(f"  - Thread Leak Delta: {thread_leak}")

    profile_summary = {
        "orchestration_construction_latency_ms": round(t_construction_ms, 2),
        "total_cold_model_load_ms": round(total_model_load_ms, 2),
        "full_cold_startup_ms": full_cold_start_ms,
        "model_breakdown_ms": model_load_metrics,
        "mean_query_latency_ms": round(mean_lat, 3),
        "median_query_latency_ms": round(median_lat, 3),
        "p95_query_latency_ms": round(p95_lat, 3),
        "max_query_latency_ms": round(max_lat, 3),
        "python_heap_peak_mb": round(peak_python_heap_mb, 2),
        "python_heap_growth_100q_kb": round(mem_growth_kb, 2),
        "total_process_rss_mb": final_process_rss_mb,
        "thread_leak_delta": thread_leak,
    }

    out_file = os.path.join(os.path.dirname(__file__), "..", "data", "profile_metrics.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(profile_summary, f, indent=2)
    print(f"\nSaved accurate Phase 3 profile metrics to {out_file}")
    return profile_summary


if __name__ == "__main__":
    run_profiling()
