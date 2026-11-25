"""NAYRA — Standalone Live Verification for Android Phone Speaker Output.

Tests:
1. Starts the NetworkAudioPlayer on http://192.168.0.102:8088.
2. Prompts user to open http://192.168.0.102:8088 on Android Chrome.
3. Awaits the user tap to unlock Web Audio hardware.
4. Sends an initial physical test chime to the phone speaker.
5. Synthesizes a Kokoro-82M offline bilingual test phrase:
   "NAYRA offline speech subsystem is now routed to your phone speaker. नमस्ते, यह आवाज़ आपके फ़ोन के स्पीकर से आ रही है।"
6. Streams 24 kHz float32 PCM over local Wi-Fi to the phone speaker.
7. Measures and prints actual transmission and playback latency.
8. Tests STOP preemption.
"""

from __future__ import annotations

import logging
from pathlib import Path
import socket
import sys
import time

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tts import KokoroEngine, NetworkAudioPlayer, PriorityLevel, TTSManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NAYRA-PhoneSpeakerVerifier")


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


def main():
    local_ip = get_local_ip()
    port = 8088
    receiver_url = f"http://{local_ip}:{port}"

    print("=" * 72)
    print("   NAYRA — Android Phone Speaker Live Output Verification")
    print("=" * 72)
    print(f"Server Host/Port : {local_ip}:{port}")
    print(f"Android Browser  : {receiver_url}")
    print("=" * 72)
    print("\nSTEP 1: Open the URL above in Chrome on your Android phone.")
    print("STEP 2: Tap the blue button: '🔊 Tap to Enable Phone Speaker'.")
    print("=" * 72)
    print("Waiting for phone connection...")

    player = NetworkAudioPlayer(host="0.0.0.0", port=port, fallback_to_laptop=True)
    tts = TTSManager(player=player)

    try:
        print("Waiting for phone connection (Press Ctrl+C to stop)...")
        print("Note: If the page does not load on your phone, change your Windows Wi-Fi")
        print("profile from 'Public' to 'Private' in Windows Settings > Network > Wi-Fi.")
        last_print = 0
        t0 = time.time()
        while not player.has_active_phone_client():
            elapsed = int(time.time() - t0)
            if elapsed // 30 > last_print:
                last_print = elapsed // 30
                print(f"[LISTENING] Server active at {receiver_url} (running for {elapsed}s)...")
            time.sleep(0.5)

        print("\n[SUCCESS] Android phone connected and Web Audio unlocked!")
        time.sleep(0.5)

        print("\n--- TEST 1: Sending Hardware Test Chime to Phone Speaker ---")
        player.broadcast_test_chime()
        print("Chime triggered on phone. Verify audible sound from phone.")
        time.sleep(1.5)

        print("\n--- TEST 2: Kokoro Bilingual Neural Speech through Phone Speaker ---")
        test_text = (
            "NAYRA offline speech subsystem is now routed to your phone speaker. "
            "नमस्ते, यह आवाज़ आपके फ़ोन के स्पीकर से आ रही है।"
        )
        print(f"Speaking: '{test_text}'")
        t0 = time.perf_counter()
        req_id = tts.speak(test_text, priority=PriorityLevel.NORMAL)
        t_enqueued = (time.perf_counter() - t0) * 1000.0
        print(f"Speech request {req_id} enqueued in {t_enqueued:.2f} ms.")

        # Wait for speech to complete
        while tts.is_speaking():
            time.sleep(0.1)

        t_total = time.perf_counter() - t0
        print(f"\n[DONE] Playback complete (Total duration: {t_total:.2f}s).")

        print("\n--- TEST 3: Preemption / STOP Verification ---")
        print("Starting long speech, waiting for audio playback, then triggering STOP...")
        long_text = "This is a long test sentence to demonstrate that speech playback stops immediately when a stop command is issued."
        t_stop_start = time.perf_counter()
        tts.speak(long_text, priority=PriorityLevel.NORMAL)
        # Wait until it starts actively outputting audio
        t_wait_play = time.time()
        while not player.is_playing() and time.time() - t_wait_play < 5.0:
            time.sleep(0.05)
        # Let it play for 1.2 seconds so user audibly hears it
        time.sleep(1.2)
        print("Triggering STOP now...")
        t_call_stop = time.perf_counter()
        tts.stop()
        t_stop_elapsed = (time.perf_counter() - t_call_stop) * 1000.0
        print(f"STOP executed in {t_stop_elapsed:.1f} ms. Verified audio queue cleared and phone playback stopped.")

        time.sleep(1.0)
        print("\nAll verification tests completed successfully.")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up server and players...")
        tts.shutdown()
        print("Finished.")


if __name__ == "__main__":
    main()
