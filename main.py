"""NAYRA — Unified Voice-First Assistive Companion MVP.

Complete software entrypoint integrating Modules 01-18:
Perception, Memory, Spatial Reasoning, Cognitive Care, Daily Reminders, Safety SOS, and Shared Voice Interaction.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from nayra_app import AppConfig, NayraApplication

logger = logging.getLogger("NAYRA-Main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NAYRA — Voice-First Assistive Companion for Visually Impaired, Elderly & Alzheimer's Care."
    )
    default_camera = os.getenv("NAYRA_STREAM_URL", "http://192.168.0.101:8080/video")
    parser.add_argument(
        "--camera",
        type=str,
        default=default_camera,
        help=f"Camera index or RTSP/HTTP stream URL (default: {default_camera}).",
    )
    parser.add_argument("--no-camera", action="store_true", help="Disable camera capture, running voice and cognitive services only.")
    parser.add_argument("--no-voice", action="store_true", help="Disable live microphone listening.")
    parser.add_argument("--phone-mic", action="store_true", help="Listen to phone microphone instead of laptop microphone.")
    parser.add_argument("--wake-word", action="store_true", default=False, help="Require 'Hey NAYRA' wake word for all queries (default: False).")
    parser.add_argument("--mock-audio", action="store_true", help="Run with mock audio instead of physical speaker output.")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display windows.")
    parser.add_argument("--text-debug", action="store_true", help="Enable interactive terminal text prompt for voice queries.")
    parser.add_argument("--lang", type=str, default="en", choices=["hi", "en"], help="Default assistant language (hi/en). Default: en.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--smoke-test", action="store_true", help="Execute an automated multi-intent smoke test and exit.")
    return parser.parse_args()


def setup_logging(level_name: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_smoke_test(app: NayraApplication) -> int:
    """Execute automated software verification test across all capabilities."""
    logger.info("=== RUNNING NAYRA UNIFIED MULTI-INTENT SMOKE TEST ===")
    test_queries = [
        ("Orientation (Alzheimer)", "Main kahan hoon?"),
        ("Obstacle Path Check", "Kya rasta saaf hai?"),
        ("Scene Description", "Mere saamne kya hai?"),
        ("Medication Reminder", "Mujhe 10 minute baad dawai lene ki yaad dilana"),
        ("List Reminders", "Mere kya reminders hain?"),
        ("Object Location Memory", "Mera phone kahan hai?"),
        ("Text Reading (OCR)", "Board par kya likha hai?"),
        ("Emergency SOS Trigger", "Madad karo emergency hai"),
        ("Emergency SOS Cancel", "Cancel kar do galti se bol diya"),
    ]

    for label, query in test_queries:
        logger.info(f"[TEST QUERY] {label}: '{query}'")
        resp = app.process_command(query)
        logger.info(f"[RESPONSE] Success={resp.success} | Spoken (HI): {resp.spoken_response_hi}")
        if not resp.success:
            logger.error(f"Smoke test failed for query: {query}")
            return 1
        time.sleep(0.05)

    logger.info("=== ALL NAYRA SMOKE TESTS PASSED SUCCESSFULLY ===")
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Starting NAYRA Unified Assistive Companion...")
    config = AppConfig(
        camera_index=args.camera,
        enable_camera=not args.no_camera,
        enable_voice_input=not args.no_voice,
        use_phone_mic=args.phone_mic,
        require_wake_word=args.wake_word,
        mock_audio=True if args.smoke_test else args.mock_audio,
        headless=args.headless,
        text_debug=args.text_debug,
        default_language=args.lang,
    )

    app = NayraApplication(config=config)

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received. Exiting gracefully...")
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.start()

        if args.smoke_test:
            code = run_smoke_test(app)
            app.shutdown()
            return code

        if args.text_debug:
            print("\n" + "=" * 60)
            print(" NAYRA Interactive Text-Voice Console")
            print(" Type voice query (or 'quit' to exit):")
            print(" Examples:")
            print("   - 'Mera phone kahan hai?'")
            print("   - 'Mere saamne kya hai?'")
            print("   - 'Kya rasta saaf hai?'")
            print("   - 'Mujhe 5 minute baad dawai ki yaad dilana'")
            print("   - 'Main kahan hoon?'")
            print("   - 'Madad karo emergency hai'")
            print("=" * 60 + "\n")

            while True:
                try:
                    user_input = input("NAYRA User > ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("quit", "exit", "q"):
                        break
                    resp = app.process_command(user_input)
                    print(f"NAYRA Spoken (HI) > {resp.spoken_response_hi}")
                    print(f"NAYRA Spoken (EN) > {resp.spoken_response_en}\n")
                except (EOFError, KeyboardInterrupt):
                    break
        else:
            logger.info("NAYRA active with Live Camera & Voice Listening enabled.")
            logger.info("Ask any question aloud (e.g. 'Where is my bottle?', 'Where is my phone?', 'Kya rasta saaf hai?').")
            logger.info("Press Ctrl+C or 'q' in preview window to terminate.")
            while not app._stop_event.is_set():
                time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        app.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
