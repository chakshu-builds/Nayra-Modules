"""Physical audio playback test script for NAYRA Module 08.

Verifies end-to-end audio output on the physical speakers/headphones via PortAudio (sounddevice).
Measures time-to-first-audio (TTFA) and confirms uninterrupted playback.
"""

from __future__ import annotations

import logging
import time

import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tts
from tts import AudioPlayer, KokoroEngine, PriorityLevel, TTSManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-PhysicalTest")


def run_physical_playback_test():
    """Execute live playback test through physical speaker/headphone hardware."""
    logger.info("Initializing TTSManager for PHYSICAL hardware playback test...")

    # Physical audio player (mock_mode=False)
    engine = KokoroEngine(use_gpu=False)
    player = AudioPlayer(mock_mode=False)
    manager = TTSManager(engine=engine, player=player, mock_audio=False)

    test_sentences = [
        ("English", "Hello, NAYRA text to speech subsystem is online and operating on local hardware.", PriorityLevel.NORMAL, "en-us"),
        ("Hindi", "नमस्ते, नयरा ध्वनि प्रणाली तैयार है।", PriorityLevel.NORMAL, "hi"),
        ("Critical", "सावधान! आगे बाधा है।", PriorityLevel.CRITICAL, "hi"),
    ]

    results = []

    for label, text, prio, lang in test_sentences:
        logger.info("Testing physical playback: [%s] '%s'...", label, text)
        t_start = time.perf_counter()

        req_id = manager.speak(text, priority=prio, language_hint=lang)

        # Poll until audio playback begins (Time to First Audio)
        ttfa = None
        while time.perf_counter() - t_start < 10.0:
            if player.is_playing():
                ttfa = (time.perf_counter() - t_start) * 1000.0
                break
            time.sleep(0.01)

        logger.info("Physical audio started for '%s' in %.2f ms (TTFA).", label, ttfa if ttfa else -1)

        # Wait until playback naturally completes
        while manager.is_speaking():
            time.sleep(0.05)

        total_time = (time.perf_counter() - t_start) * 1000.0
        logger.info("Physical playback completed for [%s] in %.2f ms.", label, total_time)

        results.append({
            "label": label,
            "text": text,
            "ttfa_ms": round(ttfa, 2) if ttfa else None,
            "total_time_ms": round(total_time, 2),
            "playback_success": ttfa is not None,
        })

        time.sleep(0.5)

    manager.shutdown()
    return results


if __name__ == "__main__":
    results = run_physical_playback_test()
    print("\n--- Physical Playback Verification Summary ---")
    for r in results:
        status = "PASSED" if r["playback_success"] else "FAILED"
        print(f"[{status}] {r['label']}: TTFA={r['ttfa_ms']} ms | Total={r['total_time_ms']} ms")
