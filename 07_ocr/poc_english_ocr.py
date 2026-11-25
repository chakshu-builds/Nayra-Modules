"""NAYRA Module 07 — Multilingual OCR Step 2: English OCR Proof-of-Concept.

Standalone script implementing:
1. English PP-OCRv5 ONNX inference (CUDA & CPU execution providers).
2. Exact aspect-ratio preserving preprocessing and dynamic padding.
3. CTC greedy decoding and confidence calculation.
4. Latency benchmarking (median and p95 across warmed-up iterations).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnglishOCR-POC")


def configure_cuda_dlls() -> bool:
    """Ensure Windows can locate CUDA/cuDNN DLLs from torch distribution if present."""
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib.resolve()))
                os.environ["PATH"] = f"{torch_lib.resolve()};{os.environ.get('PATH', '')}"
            except (AttributeError, OSError):
                pass
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def load_character_dict(yml_path: str | Path) -> List[str]:
    """Parse character dictionary from English PP-OCRv5 YAML config.
    
    Vocabulary format:
      Index 0: 'blank'
      Index 1..N: Characters from character_dict
      Index N+1: ' ' (space character)
    """
    path = Path(yml_path)
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    in_dict = False
    dict_chars = []

    for line in lines:
        if line.strip().startswith("character_dict:"):
            in_dict = True
            continue
        if in_dict:
            if line.startswith("  - "):
                val = line[4:]
                if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                    val = val[1:-1]
                dict_chars.append(val)
            elif not line.startswith(" "):
                break

    character_list = ["blank"] + dict_chars + [" "]
    logger.info(f"Loaded English dictionary with {len(character_list)} tokens ({len(dict_chars)} chars + blank + space)")
    return character_list


def preprocess_image(img: np.ndarray, target_h: int = 48, min_w: int = 320) -> Tuple[np.ndarray, float]:
    """Preprocess text crop image for PP-OCRv5 recognition.
    
    - Resizes to height target_h maintaining aspect ratio.
    - Pads width to next multiple of 32 (at least min_w).
    - Normalizes: (x / 255.0 - 0.5) / 0.5.
    - Zero-pads the canvas to pad_w.
    
    Returns:
      tensor: (1, 3, target_h, pad_w) float32
      valid_ratio: ratio of real text width to pad_w
    """
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = int(round(target_h * ratio))
    
    pad_w = max(min_w, int(np.ceil(resized_w / 32) * 32))
    resized = cv2.resize(img, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    canvas = np.zeros((3, target_h, pad_w), dtype=np.float32)
    canvas[:, :, :resized_w] = norm.transpose((2, 0, 1))
    
    tensor = canvas[np.newaxis, ...]
    valid_ratio = min(1.0, resized_w / float(pad_w))
    return tensor, valid_ratio


def ctc_decode(
    preds_prob: np.ndarray,
    character_list: List[str],
    valid_ratio: float = 1.0,
) -> Tuple[str, float, List[Tuple[int, str, float]]]:
    """Decode English PP-OCRv5 CTC output probabilities."""
    T = preds_prob.shape[0]
    valid_timesteps = int(round(T * valid_ratio))
    
    preds_idx = np.argmax(preds_prob, axis=-1)
    max_probs = np.max(preds_prob, axis=-1)
    
    decoded_chars = []
    conf_scores = []
    char_details = []
    prev_idx = -1
    
    for t in range(valid_timesteps):
        idx = preds_idx[t]
        if idx != 0 and idx != prev_idx:
            char = character_list[idx]
            conf = float(max_probs[t])
            decoded_chars.append(char)
            conf_scores.append(conf)
            char_details.append((t, char, conf))
        prev_idx = idx

    text = "".join(decoded_chars)
    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
    return text, avg_conf, char_details


class EnglishOCRModel:
    """English PP-OCRv5 standalone ONNX recognition wrapper."""

    def __init__(
        self,
        model_path: str | Path = "models/ocr/en_PP-OCRv5_mobile_rec.onnx",
        dict_path: str | Path = "models/ocr/en_PP-OCRv5_mobile_rec.yml",
        use_cuda: bool = True,
    ):
        self.model_path = Path(model_path)
        self.dict_path = Path(dict_path)
        self.character_list = load_character_dict(self.dict_path)

        if use_cuda:
            configure_cuda_dlls()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.active_providers = self.session.get_providers()
        logger.info(f"Initialized English ONNX session with providers: {self.active_providers}")

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """Run recognition on a single text line image."""
        tensor, valid_ratio = preprocess_image(img)
        outputs = self.session.run(None, {"x": tensor})[0]
        text, conf, _ = ctc_decode(outputs[0], self.character_list, valid_ratio)
        return text, conf

    def benchmark_latency(self, img: np.ndarray, warmup: int = 10, iterations: int = 50) -> Dict[str, float]:
        """Benchmark forward pass latency in milliseconds."""
        tensor, _ = preprocess_image(img)
        
        # Warmup
        for _ in range(warmup):
            _ = self.session.run(None, {"x": tensor})
            
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self.session.run(None, {"x": tensor})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
        latencies.sort()
        median_ms = float(np.median(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        
        return {
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "iterations": iterations,
        }


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 07 English OCR POC")
    parser.add_argument("--image", type=str, required=True, help="Path to text image crop")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    use_cuda = args.device == "cuda"
    model = EnglishOCRModel(use_cuda=use_cuda)

    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Failed to read image: {args.image}")
        sys.exit(1)

    text, conf = model.predict(img)
    print("\n" + "=" * 50)
    print(f"Image           : {args.image}")
    print(f"Recognized Text : {text}")
    print(f"Confidence      : {conf:.4f}")
    print("=" * 50)

    if args.benchmark:
        bench = model.benchmark_latency(img, warmup=10, iterations=50)
        print(f"Latency (median): {bench['median_ms']} ms")
        print(f"Latency (p95)   : {bench['p95_ms']} ms")
        print(f"Latency (min)   : {bench['min_ms']} ms")
        print(f"Latency (max)   : {bench['max_ms']} ms")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
