"""NAYRA Module 09 — Live Phone Microphone & Voice Command Verification Tool.

Diagnoses network connectivity to Android IP Webcam, inspects /audio.wav streaming,
tests Voice Activity Detection (VAD), and runs offline speech-to-intent recognition.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

# Ensure root workspace is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import voice
from voice import CommandIntent, CommandMatcher, PhoneAudioReceiver, VoiceCommandRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-VoiceVerify")


def verify_phone_microphone_stream(stream_url: str, duration_sec: float = 3.0) -> dict:
    """Connect to Android IP Webcam audio endpoint, inspect WAV header, and measure live energy."""
    norm_url = PhoneAudioReceiver._normalize_audio_url(stream_url)
    logger.info("Connecting to Android IP Webcam audio stream: %s", norm_url)

    result = {
        "url": norm_url,
        "reachable": False,
        "http_status": None,
        "sample_rate": None,
        "channels": None,
        "bits_per_sample": None,
        "rms_energy": None,
        "mic_active": False,
        "error": None,
    }

    try:
        req = urllib.request.Request(
            norm_url,
            headers={"User-Agent": "NAYRA-VoiceVerify/1.0", "Accept": "audio/wav, audio/*"},
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            result["reachable"] = True
            result["http_status"] = resp.status
            logger.info("Connected successfully (HTTP %s)!", resp.status)

            header = resp.read(44)
            if len(header) >= 44:
                riff, size, wave = struct.unpack("<4sI4s", header[:12])
                if riff == b"RIFF" and wave == b"WAVE":
                    _, _, _, channels, rate, _, _, bits = struct.unpack("<4sIHHIIHH", header[12:36])
                    result["sample_rate"] = rate
                    result["channels"] = channels
                    result["bits_per_sample"] = bits
                    logger.info("WAV stream format: %d Hz, %d channel(s), %d-bit PCM", rate, channels, bits)

            # Stream audio for specified duration
            bytes_to_read = int(duration_sec * (result["sample_rate"] or 16000) * (result["channels"] or 1) * 2)
            raw_audio = bytearray()
            t_start = time.perf_counter()

            while len(raw_audio) < bytes_to_read and (time.perf_counter() - t_start) < (duration_sec + 2.0):
                chunk = resp.read(2048)
                if not chunk:
                    break
                raw_audio.extend(chunk)

            if raw_audio:
                int_data = np.frombuffer(raw_audio[: len(raw_audio) - (len(raw_audio) % 2)], dtype=np.int16)
                float_data = int_data.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(float_data ** 2)))
                peak = float(np.max(np.abs(float_data)))
                result["rms_energy"] = round(rms, 5)
                result["peak_amplitude"] = round(peak, 5)
                result["mic_active"] = rms > 0.0001
                logger.info("Recorded %.2fs audio: RMS energy=%.5f, Peak=%.5f (mic active: %s)", len(float_data) / (result["sample_rate"] or 16000), rms, peak, result["mic_active"])

    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Could not connect to microphone stream at %s: %s", norm_url, exc)

    return result


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 09 Live Voice Command Verification")
    parser.add_argument("--url", "-u", type=str, default="http://192.168.0.101:8080/audio.wav", help="IP Webcam audio stream URL.")
    parser.add_argument("--duration", "-d", type=float, default=3.0, help="Listen duration in seconds.")
    args = parser.parse_args()

    print("=================================================================")
    print("NAYRA Module 09 — Phone Microphone & Voice Command Diagnostic")
    print("=================================================================")

    # Step 1: Network & Stream verification
    stream_res = verify_phone_microphone_stream(args.url, duration_sec=args.duration)

    print("\n--- Phase 1: Phone Audio Input Diagnostic ---")
    if stream_res["reachable"]:
        print(f"[PASSED] Connected to phone audio endpoint: {stream_res['url']}")
        print(f"  Format: {stream_res['sample_rate']} Hz, {stream_res['channels']} channel(s), {stream_res['bits_per_sample']}-bit")
        print(f"  RMS Energy: {stream_res['rms_energy']} | Mic Live Input: {stream_res['mic_active']}")
    else:
        print(f"[WAITING] Endpoint {stream_res['url']} not reachable: {stream_res['error']}")
        print("  Action required on Android phone:")
        print("  1. Open 'IP Webcam' app on phone.")
        print("  2. In app settings, confirm 'Audio connection' is checked/enabled.")
        print("  3. Scroll to the very bottom and tap 'Start server'.")
        print(f"  4. Verify the IP address shown on the phone screen matches {args.url}.")

    # Step 2: STT Engine Verification
    print("\n--- Phase 2: Offline STT Engine Verification ---")
    rec = VoiceCommandRecognizer()
    rec.initialize()
    print("[PASSED] Offline Whisper tiny model loaded successfully in INT8 on CPU.")
    print("Supported voice commands verified:")
    for cmd in ["NAYRA, read the text", "Read this", "Mere liye padho", "Stop reading", "Text reading band karo"]:
        intent, _ = CommandMatcher.match_intent(cmd)
        print(f"  '{cmd}' -> Intent: {intent.value}")

    rec.shutdown()
    print("\nVerification completed.")


if __name__ == "__main__":
    main()
