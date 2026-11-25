"""NAYRA Phase 3 — End-to-End Voice-Controlled Read Aloud Runner.

Connects to:
1. Android IP Webcam microphone stream (http://<phone_ip>:8080/audio.wav)
2. Android IP Webcam camera stream (http://<phone_ip>:8080/video) or test frame
3. Module 07 Hindi-English OCR Engine
4. Module 08 Kokoro-82M Offline TTS Engine
5. Module 09 Whisper-tiny INT8 Speech Command Recognizer

Usage:
    python 09_voice/run_read_aloud.py --ip 192.168.0.101
    python 09_voice/run_read_aloud.py --simulate
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Set OpenCV / FFmpeg network timeout options to prevent stream hangs
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;5000000|stimeout;5000000",
)

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from voice import CommandIntent, PhoneAudioReceiver, ReadAloudPipeline, VoiceCommandRecognizer
from tts import PriorityLevel, TTSManager
import ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NAYRA-ReadAloudRunner")


def build_frame_provider(video_url: Optional[str], test_image_path: Optional[Path]):
    """Construct a resilient camera frame provider with fresh on-demand capture."""
    fallback_frame = None
    if test_image_path and test_image_path.exists():
        fallback_frame = cv2.imread(str(test_image_path))
        logger.info("Loaded fallback test frame from %s (%s)", test_image_path, fallback_frame.shape)

    def _get_frame() -> Optional[np.ndarray]:
        if video_url:
            cap = cv2.VideoCapture(video_url)
            if cap.isOpened():
                frame = None
                try:
                    # Flush initial sensor warmup frames (up to 8 frames) until exposure settles
                    for _ in range(8):
                        ret, tmp = cap.read()
                        if ret and tmp is not None and tmp.size > 0:
                            frame = tmp
                            # If sensor has adjusted exposure and is not pitch black, take it
                            if np.mean(frame) >= 10.0:
                                break
                    if frame is not None and frame.size > 0:
                        logger.info(
                            "Acquired fresh camera frame: %s (mean brightness=%.1f)",
                            frame.shape,
                            float(np.mean(frame)),
                        )
                        return frame
                except Exception as exc:
                    logger.warning("Error reading from video stream %s: %s", video_url, exc)
                finally:
                    cap.release()
            else:
                logger.warning("Could not connect to video stream %s; falling back to test image.", video_url)
        return fallback_frame

    return _get_frame, None


def main():
    parser = argparse.ArgumentParser(description="NAYRA Voice-Controlled Read Aloud Runner")
    parser.add_argument("--ip", type=str, default="192.168.0.101", help="Android device IP address")
    parser.add_argument("--audio-port", type=int, default=8080, help="IP Webcam port (default: 8080)")
    parser.add_argument("--debounce", type=float, default=2.0, help="Debounce interval in seconds")
    parser.add_argument("--test-image", type=str, default="image1.jpg", help="Path to static image fallback")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without requiring live phone")
    parser.add_argument("--mock-audio", action="store_true", help="Do not output physical audio to speakers")
    parser.add_argument("--ocr-device", type=str, default="CPU", choices=["CUDA", "CPU"], help="Execution device for Module 07 OCR (default: CPU)")
    parser.add_argument("--phone-speaker", action="store_true", default=False, help="Route TTS audio to Android phone speaker over Wi-Fi")
    parser.add_argument("--phone-audio-port", type=int, default=8088, help="Port for Android phone audio receiver web page (default: 8088)")
    parser.add_argument("--receiver-host", type=str, default="0.0.0.0", help="Bind address for phone audio receiver (default: 0.0.0.0)")
    parser.add_argument("--speaker-output", type=str, choices=["phone", "laptop"], default="laptop", help="Audio output destination ('laptop' or 'phone') (default: laptop)")
    parser.add_argument("--wait-for-phone", type=float, default=0.0, help="Wait up to N seconds for phone receiver to connect before starting (default: 0)")

    args = parser.parse_args()

    use_phone_speaker = (args.speaker_output == "phone") or (args.phone_speaker and args.speaker_output != "laptop")

    audio_url = f"http://{args.ip}:{args.audio_port}/audio.wav"
    video_url = f"http://{args.ip}:{args.audio_port}/video" if not args.simulate else None
    test_img = root_dir / args.test_image

    print("=" * 72)
    print("   NAYRA — End-to-End Voice-Controlled Read Aloud Subsystem")
    print("=" * 72)
    print(f"Mode              : {'SIMULATION' if args.simulate else 'LIVE NETWORK'}")
    print(f"Phone IP          : {args.ip}:{args.audio_port}")
    print(f"Audio Endpoint    : {audio_url}")
    print(f"Video Endpoint    : {video_url or 'Static Test Image (' + args.test_image + ')'}")
    if use_phone_speaker:
        import socket
        def get_local_ip() -> str:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip
        phone_receiver_url = f"http://{get_local_ip()}:{args.phone_audio_port}"
        print(f"Audio Output      : PHONE SPEAKER -> {phone_receiver_url}")
    else:
        print(f"Audio Output      : LAPTOP SPEAKERS (PortAudio Local Hardware)")
    print(f"Debounce Window   : {args.debounce:.1f}s")
    print("=" * 72)
    print("Supported Voice Commands:")
    print("  [READ] : 'NAYRA, read the text' | 'Read this' | 'मेरे लिए पढ़ो'")
    print("  [STOP] : 'Stop reading' | 'Text reading band karo' | 'रोक दो'")
    print("-" * 72)
    print("Keyboard Controls (Terminal):")
    print("  [R] + Enter : Manually trigger OCR Read")
    print("  [S] + Enter : Manually trigger Stop Speech")
    print("  [Q] + Enter : Quit")
    print("=" * 72)

    frame_provider, _ = build_frame_provider(video_url, test_img)

    tts_mgr = TTSManager(
        mock_audio=args.mock_audio,
        phone_speaker=use_phone_speaker,
        phone_audio_port=args.phone_audio_port,
    )
    recognizer = VoiceCommandRecognizer()

    if use_phone_speaker and args.wait_for_phone > 0:
        print(f"Waiting up to {args.wait_for_phone:.0f}s for phone receiver to connect at {phone_receiver_url}...")
        t_wait_start = time.time()
        connected = False
        while time.time() - t_wait_start < args.wait_for_phone:
            if hasattr(tts_mgr.player, "has_active_phone_client") and tts_mgr.player.has_active_phone_client():
                connected = True
                print("[CONNECTED] Android phone speaker receiver connected and active!")
                break
            time.sleep(0.5)
        if not connected:
            print("[NOTICE] Phone receiver not yet active; pipeline starting with laptop speaker fallback.")

    def telemetry_callback(event_name: str, data: dict):
        if event_name == "speech_recognized":
            print(f"\n[VOICE] >>> '{data['transcript']}' -> INTENT: {data['intent']} ({data['latency_ms']:.1f} ms)")
        elif event_name == "ocr_completed":
            providers = data.get("execution_providers", {})
            print(f"[OCR]   >>> Success={data['success']} | Language={data['primary_language']} | Latency={data['ocr_latency_ms']:.1f} ms | Providers={providers}")
        elif event_name == "tts_enqueued":
            print(f"[TTS]   >>> Speaking ({data['request_id']}): {data['text']}")
        elif event_name == "command_read_debounced":
            print(f"[GATE]  >>> Duplicate read ignored (cooldown: {data['elapsed_sec']:.2f}s)")

    receiver = PhoneAudioReceiver(
        stream_url=audio_url,
        mock_mode=args.simulate,
    )

    pipeline = ReadAloudPipeline(
        audio_receiver=receiver,
        voice_recognizer=recognizer,
        tts_manager=tts_mgr,
        frame_provider=frame_provider,
        preferred_ocr_provider=args.ocr_device,
        debounce_sec=args.debounce,
        mock_mode=args.simulate,
        on_event=telemetry_callback,
    )

    pipeline.start()

    # Initial announcement
    tts_mgr.speak("NAYRA Read Aloud subsystem is ready.", priority=PriorityLevel.NORMAL)

    print("\nListening for voice commands. Press Q and Enter to quit...")
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == "q":
                break
            elif cmd == "r":
                print("[MANUAL TRIGGER] Executing manual OCR read...")
                frame = frame_provider()
                if frame is not None:
                    res = ocr.read_text(frame, preferred_provider=args.ocr_device)
                    if res.combined_text:
                        print(f"[OCR RESULT] {res.combined_text}")
                        tts_mgr.speak(res.combined_text, priority=PriorityLevel.NORMAL, source="ocr")
                    else:
                        tts_mgr.speak("No text detected.", priority=PriorityLevel.NORMAL, source="ocr")
            elif cmd == "s":
                print("[MANUAL TRIGGER] Halting TTS playback...")
                tts_mgr.stop()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nShutting down NAYRA Read Aloud pipeline...")
        pipeline.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
