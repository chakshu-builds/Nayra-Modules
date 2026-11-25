import os
import sys
import time
from pathlib import Path
import numpy as np

# Configure CUDA DLLs from torch
try:
    import torch
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib.resolve()))
            os.environ["PATH"] = f"{torch_lib.resolve()};{os.environ.get('PATH', '')}"
        except (AttributeError, OSError):
            pass
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not found")

import onnxruntime as ort

print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

models = {
    'ch_PP-OCRv4_det_deepghs': '07_ocr/audit_models/ch_PP-OCRv4_det_deepghs.onnx',
    'ch_PP-OCRv4_det_tobiichi': '07_ocr/audit_models/ch_PP-OCRv4_det_tobiichi.onnx',
    'PP-OCRv5_mobile_det': '07_ocr/audit_models/PP-OCRv5_mobile_det.onnx'
}

resolutions = [(640, 640), (960, 960)]
providers_list = [
    ('CUDA', ['CUDAExecutionProvider', 'CPUExecutionProvider']),
    ('CPU', ['CPUExecutionProvider'])
]

results = []

for mname, mpath in models.items():
    if not Path(mpath).exists():
        print(f"Skipping {mname}, not found.")
        continue
    print(f"\n==========================================")
    print(f"MODEL: {mname}")
    print(f"==========================================")
    for prov_label, prov in providers_list:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(mpath, sess_options=sess_opt, providers=prov)
            active_provs = sess.get_providers()
            print(f"\n--- Provider [{prov_label}] Requested, Active: {active_provs} ---")
            
            input_info = sess.get_inputs()[0]
            output_info = sess.get_outputs()[0]
            input_name = input_info.name
            output_name = output_info.name
            print(f"Input: {input_name} ({input_info.shape}), Output: {output_name} ({output_info.shape})")
            
            for h, w in resolutions:
                dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)
                
                # Warmup (5 runs)
                for _ in range(5):
                    _ = sess.run([output_name], {input_name: dummy_input})
                
                # Benchmark (20 iterations)
                latencies = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    out = sess.run([output_name], {input_name: dummy_input})
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                
                med_lat = np.median(latencies)
                p95_lat = np.percentile(latencies, 95)
                min_lat = np.min(latencies)
                max_lat = np.max(latencies)
                
                print(f"  Res {h}x{w} ({prov_label}) -> Median: {med_lat:6.2f} ms | P95: {p95_lat:6.2f} ms | Min: {min_lat:6.2f} ms | Max: {max_lat:6.2f} ms | Out shape: {out[0].shape}")
                
                results.append({
                    "model": mname,
                    "provider": prov_label,
                    "active_providers": active_provs,
                    "resolution": f"{h}x{w}",
                    "median_ms": round(float(med_lat), 2),
                    "p95_ms": round(float(p95_lat), 2),
                    "min_ms": round(float(min_lat), 2),
                    "max_ms": round(float(max_lat), 2),
                })
        except Exception as e:
            print(f"  Provider [{prov_label}] Error: {e}")

import json
out_path = Path("07_ocr/audit_models/det_benchmark_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved benchmark results to {out_path}")
