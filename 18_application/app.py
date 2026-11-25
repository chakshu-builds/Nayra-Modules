"""Unified NAYRA Assistive Application coordinating all modules and interfaces."""

from __future__ import annotations

import logging
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Optional
import cv2
import numpy as np

from shared.schemas import IntentResponse, SpeechPriority, UserIntent
from voice_interaction import VoiceManager
from assistant_orchestrator import AssistantOrchestrator
from obstacle_awareness import ObstacleService
from scene_understanding import SceneService
from alzheimer_assistance import AlzheimerService
from daily_assistance import DailyService
from safety_sos import SafetyService

try:
    from .adapters import ObjectMemoryAdapter, OCRAdapter
    from .app_config import AppConfig
except (ImportError, ValueError):
    from adapters import ObjectMemoryAdapter, OCRAdapter
    from app_config import AppConfig

logger = logging.getLogger("NAYRA-Application")


class NayraApplication:
    """Central unified application lifecycle coordinator for NAYRA."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

        # Physical TTS audio backend (Kokoro-82M on Laptop Speakers)
        tts_backend = None
        if not getattr(self.config, "mock_audio", True):
            try:
                from tts import TTSManager
                tts_backend = TTSManager(phone_speaker=False, mock_audio=False)
                logger.info("Physical Kokoro TTS backend initialized for laptop speaker playback.")
            except Exception as tts_err:
                logger.warning(f"Could not initialize physical TTSManager: {tts_err}. Continuing with simulation.")
                tts_backend = None

        self.voice_manager = VoiceManager(tts_backend=tts_backend)

        classifier = None
        try:
            from assistant_orchestrator.intent_router import IntentClassifier
            classifier = IntentClassifier(require_wake_word=getattr(self.config, "require_wake_word", False))
        except Exception:
            classifier = None
        pref_lang = self.config.default_language
        self.orchestrator = AssistantOrchestrator(
            voice_output=self.voice_manager,
            classifier=classifier,
            preferred_language=pref_lang,
        )

        # Domain Services
        self.obstacle_service = ObstacleService(voice_output=self.voice_manager, preferred_language=pref_lang)
        self.scene_service = SceneService(voice_output=self.voice_manager, preferred_language=pref_lang)
        self.alzheimer_service = AlzheimerService(voice_output=self.voice_manager, preferred_language=pref_lang)

        # Reminder DB configuration (allows isolated temporary or custom demo databases)
        if getattr(self.config, "reminders_db_path", None):
            try:
                from daily_assistance.reminder_db import ReminderDB
                db = ReminderDB(db_path=Path(self.config.reminders_db_path))
                self.daily_service = DailyService(voice_output=self.voice_manager, db=db, preferred_language=pref_lang)
            except Exception as e:
                logger.warning(f"Could not initialize custom ReminderDB: {e}")
                self.daily_service = DailyService(voice_output=self.voice_manager, preferred_language=pref_lang)
        else:
            self.daily_service = DailyService(voice_output=self.voice_manager, preferred_language=pref_lang)

        self.safety_service = SafetyService(voice_output=self.voice_manager, preferred_language=pref_lang)

        # Object memory manager integration
        memory_manager = None
        try:
            from object_memory import ObjectMemoryManager
            om_db = Path(self.config.object_memory_db_path) if getattr(self.config, "object_memory_db_path", None) else None
            memory_manager = ObjectMemoryManager(db_path=om_db)
            logger.info("ObjectMemoryManager initialized in NayraApplication.")
        except Exception as e:
            logger.warning(f"Could not initialize ObjectMemoryManager in NayraApplication: {e}")

        self.object_memory_adapter = ObjectMemoryAdapter(
            memory_manager=memory_manager,
            voice_output=self.voice_manager,
            is_camera_active=self.config.enable_camera,
            preferred_language=pref_lang,
        )
        self.ocr_adapter = OCRAdapter(
            voice_output=self.voice_manager,
            preferred_language=pref_lang,
            frame_provider=lambda: self._latest_camera_frame,
        )

        self._register_modules()
        self._running = False
        self._initialized = False
        self._stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._audio_receiver = None
        self._latest_camera_frame: Optional[np.ndarray] = None
        self._latest_user_speech: str = ""
        self._latest_assistant_reply: str = ""
        self._latest_speech_time: float = 0.0

    def _register_modules(self) -> None:
        """Register all domain modules into the central orchestrator."""
        self.orchestrator.register_module("11_obstacle_awareness", self.obstacle_service)
        self.orchestrator.register_module("12_scene_understanding", self.scene_service)
        self.orchestrator.register_module("13_alzheimer_assistance", self.alzheimer_service)
        self.orchestrator.register_module("14_daily_assistance", self.daily_service)
        self.orchestrator.register_module("15_safety_sos", self.safety_service)
        self.orchestrator.register_module("10_object_memory", self.object_memory_adapter)
        self.orchestrator.register_module("07_ocr", self.ocr_adapter)

    def initialize(self) -> bool:
        """Initialize voice subsystems and all registered modules (idempotent)."""
        if self._initialized:
            return True
        logger.info("Initializing NAYRA Application subsystems...")
        self.voice_manager.start()
        ok = self.orchestrator.initialize_all()
        if ok:
            logger.info("All NAYRA domain modules initialized successfully.")
        else:
            logger.warning("Some NAYRA modules reported initialization warnings.")
        self._initialized = True
        return ok

    def start(self) -> None:
        """Start the application and announce readiness."""
        self.initialize()
        self._running = True

        # Start camera perception thread if camera is enabled
        if self.config.enable_camera:
            self._camera_thread = threading.Thread(
                target=self._camera_perception_loop,
                name="NAYRA-CameraPerception",
                daemon=True,
            )
            self._camera_thread.start()
            logger.info("Camera perception thread started.")

        # Start live voice listening thread if voice input is enabled
        if getattr(self.config, "enable_voice_input", False):
            self._audio_thread = threading.Thread(
                target=self._voice_listener_loop,
                name="NAYRA-VoiceListener",
                daemon=True,
            )
            self._audio_thread.start()
            logger.info("Live voice listening thread started.")

        # Spoken greeting
        greeting_hi = "NAYRA assistive companion taiyaar hai. Main aapki kya madad kar sakti hoon?"
        greeting_en = "NAYRA assistive companion is ready. How may I help you?"
        chosen_greeting = greeting_hi if self.config.default_language == "hi" else greeting_en

        self.voice_manager.speak(chosen_greeting, priority=SpeechPriority.INFORMATIONAL)
        logger.info(f"NAYRA started: {chosen_greeting}")

    def _camera_perception_loop(self) -> None:
        """Background perception thread capturing frames, detecting objects, and updating spatial context."""
        try:
            import cv2
            from object_detection import ObjectDetector
        except ImportError as e:
            logger.warning(f"Computer vision dependencies unavailable: {e}. Camera loop skipped.")
            return

        cam_src = int(self.config.camera_index) if str(self.config.camera_index).isdigit() else self.config.camera_index
        logger.info(f"Connecting to camera source: {cam_src}")

        # Attempt high-performance zero-lag threaded stream (Android MJPEG / RTSP / local webcam)
        stream = None
        cap = None
        try:
            import importlib
            _mod = importlib.import_module("10_object_memory.perception.live_camera_stream")
            LiveCameraStream = getattr(_mod, "LiveCameraStream", None)
            if LiveCameraStream is not None:
                stream = LiveCameraStream(source=cam_src, target_fps=30.0, connection_timeout_sec=3.0)
                if not stream.start():
                    logger.warning(f"LiveCameraStream could not connect to '{cam_src}'. Operating in voice-only mode.")
                    return
                logger.info(f"High-performance zero-lag LiveCameraStream connected to: {cam_src}")
        except Exception as stream_err:
            logger.debug(f"LiveCameraStream init skipped: {stream_err}. Using standard capture.")
            stream = None

        if stream is None:
            cap = cv2.VideoCapture(cam_src)
            if not cap.isOpened():
                logger.warning(f"Could not open camera '{cam_src}'. Operating in voice-only assistive mode.")
                return
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        memory_mgr = getattr(self.object_memory_adapter, "memory", None)
        detector = None
        tracker = None

        # If memory manager is not available, initialize fallback detector and tracker
        if not (memory_mgr and hasattr(memory_mgr, "observe_frame")):
            try:
                detector = ObjectDetector()
                logger.info("Fallback ObjectDetector initialized for live camera perception.")
            except Exception as e:
                logger.warning(f"Could not initialize ObjectDetector: {e}. Running camera in passthrough mode.")

            try:
                from object_tracking import ByteTracker
                tracker = ByteTracker()
                logger.info("Fallback ByteTracker initialized for live obstacle tracking.")
            except Exception as e:
                logger.warning(f"Could not initialize ByteTracker in camera loop: {e}")

        frame_counter = 0
        sample_interval = 2  # Sample YOLO every 2 frames to sustain 25-30 FPS display
        detections = []
        tracks = []
        t_prev_fps = time.perf_counter()
        display_fps = 0.0

        try:
            while not self._stop_event.is_set():
                if stream is not None:
                    ret, frame, ts = stream.read_latest()
                    if not ret or frame is None or frame.size == 0:
                        time.sleep(0.005)
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        time.sleep(0.01)
                        continue
                    ts = time.time()

                self._latest_camera_frame = frame
                frame_counter += 1
                h, w = frame.shape[:2]

                # FPS Calculation
                t_now = time.perf_counter()
                dt = t_now - t_prev_fps
                t_prev_fps = t_now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    display_fps = inst_fps if display_fps == 0.0 else (0.15 * inst_fps + 0.85 * display_fps)

                # Execute perception (YOLO + Tracker) on cadence without duplicate forward passes
                if frame_counter % sample_interval == 0:
                    if memory_mgr and hasattr(memory_mgr, "observe_frame"):
                        try:
                            memory_mgr.observe_frame(frame, timestamp=ts, force_detection=False)
                            detections = getattr(memory_mgr, "_last_frame_detections", []) or []
                            if getattr(memory_mgr, "tracker", None):
                                all_t = getattr(memory_mgr.tracker, "tracked_tracks", [])
                                tracks = [t for t in all_t if getattr(t, "consecutive_missed_frames", 0) == 0]
                        except Exception as mem_err:
                            logger.debug(f"Memory perception error: {mem_err}")
                    elif detector is not None:
                        try:
                            dets, _ = detector.detect(frame)
                            detections = dets or []
                            if tracker is not None and detections:
                                tracks, _ = tracker.update(detections)
                        except Exception as det_err:
                            logger.debug(f"Inference error in camera loop: {det_err}")

                    # Update live spatial and obstacle modules
                    try:
                        self.obstacle_service.update_obstacles(
                            detections, tracks=tracks, frame_width=w, frame_height=h, auto_speak=True
                        )
                        self.scene_service.update_scene_context(detections, frame_width=w, frame_height=h)
                    except Exception as update_err:
                        logger.debug(f"Perception state update error: {update_err}")

                # Live Camera Preview Window on Laptop Screen
                if not self.config.headless:
                    try:
                        vis = frame.copy()
                        active_tracker = getattr(memory_mgr, "tracker", None) or tracker
                        active_detector = getattr(memory_mgr, "detector", None) or detector

                        if active_tracker is not None and tracks and hasattr(active_tracker, "draw_tracks"):
                            vis = active_tracker.draw_tracks(vis, tracks, draw_hud=False)
                        elif active_detector is not None and detections and hasattr(active_detector, "draw_detections"):
                            vis = active_detector.draw_detections(vis, detections, draw_hud=False)
                        elif detections:
                            for d in detections:
                                bx1, by1, bx2, by2 = [int(v) for v in d.box_xyxy]
                                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                                label = f"{d.class_name.upper()} {int(d.confidence * 100)}%"
                                cv2.putText(vis, label, (bx1, max(20, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Overlay assistive telemetry HUD
                        hud_text = f"NAYRA Live | {display_fps:.1f} FPS | Objects: {len(detections)} | Tracks: {len(tracks)}"
                        hud_w = min(w - 20, 520)
                        cv2.rectangle(vis, (10, 10), (10 + hud_w, 42), (20, 20, 20), -1)
                        cv2.rectangle(vis, (10, 10), (10 + hud_w, 42), (0, 255, 0), 1)
                        cv2.putText(vis, hud_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # Overlay Navigational Path Guidance
                        hazards = getattr(self.obstacle_service, "_latest_hazards", []) or []
                        center_hazards = [hz for hz in hazards if hz.zone == "CENTER" and hz.proximity in ("NEAR", "MEDIUM")]
                        urgent_hazards = [hz for hz in hazards if hz.is_urgent]

                        b_left = (int(w * 0.22), h - 15)
                        b_right = (int(w * 0.78), h - 15)
                        m_left = (int(w * 0.38), int(h * 0.52))
                        m_right = (int(w * 0.62), int(h * 0.52))

                        if center_hazards or urgent_hazards:
                            p_color = (0, 0, 255)  # RED warning
                            obs_name = (center_hazards[0].class_name if center_hazards else urgent_hazards[0].class_name).upper()
                            p_status = f"WARNING: {obs_name} IN PATH"
                        else:
                            p_color = (0, 255, 0)  # GREEN clear
                            p_status = "NAV PATH: CLEAR"

                        corridor_pts = np.array([b_left, m_left, m_right, b_right], dtype=np.int32)
                        poly_overlay = vis.copy()
                        cv2.fillPoly(poly_overlay, [corridor_pts], p_color)
                        cv2.addWeighted(poly_overlay, 0.15, vis, 0.85, 0, vis)
                        cv2.polylines(vis, [corridor_pts], isClosed=True, color=p_color, thickness=2)

                        # Status badge for path
                        badge_x1, badge_y1 = int(w * 0.25), int(h * 0.52) - 26
                        badge_x2, badge_y2 = int(w * 0.75), int(h * 0.52)
                        cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 15, 15), -1)
                        cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), p_color, 2)
                        cv2.putText(vis, p_status, (badge_x1 + 12, badge_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, p_color, 2)

                        # Overlay speech subtitles (User question & NAYRA answer) on screen
                        if self._latest_user_speech and (time.time() - self._latest_speech_time < 12.0):
                            box_h = 75 if self._latest_assistant_reply else 44
                            box_y = max(10, h - box_h - 15)
                            # Dark background banner with cyan border
                            cv2.rectangle(vis, (15, box_y), (w - 15, box_y + box_h), (15, 15, 15), -1)
                            cv2.rectangle(vis, (15, box_y), (w - 15, box_y + box_h), (0, 220, 255), 2)
                            # User question line
                            user_txt = f"YOU: {self._latest_user_speech}"
                            if len(user_txt) > 55:
                                user_txt = user_txt[:52] + "..."
                            cv2.putText(vis, user_txt, (25, box_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                            # NAYRA answer line
                            if self._latest_assistant_reply:
                                ans_txt = f"NAYRA: {self._latest_assistant_reply}"
                                if len(ans_txt) > 60:
                                    ans_txt = ans_txt[:57] + "..."
                                cv2.putText(vis, ans_txt, (25, box_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 255, 120), 2)

                        window_title = "NAYRA — Live Camera Preview (Press 'q' to quit)"
                        cv2.imshow(window_title, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            logger.info("Quit requested via camera preview window.")
                            self._stop_event.set()
                            break
                    except Exception as preview_err:
                        logger.debug(f"Live preview rendering error: {preview_err}")

                # Yield thread briefly to prevent CPU starvation
                time.sleep(0.002)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if not self.config.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            logger.info("Camera capture released.")

    def process_command(self, query: str) -> IntentResponse:
        """Dispatch a natural speech query to the assistant orchestrator."""
        self._latest_user_speech = query
        self._latest_speech_time = time.time()
        resp = self.orchestrator.process_voice_query(query)
        chosen_reply = resp.spoken_response_en if self.config.default_language == "en" else (resp.spoken_response_hi or resp.spoken_response_en)
        self._latest_assistant_reply = chosen_reply or ""
        self._latest_speech_time = time.time()
        return resp

    def _voice_listener_loop(self) -> None:
        """Background live voice listening worker transcribing speech and routing voice queries."""
        try:
            import numpy as np
            from voice import PhoneAudioReceiver, VoiceCommandRecognizer
        except ImportError as exc:
            logger.warning(f"Voice recognition dependencies unavailable: {exc}. Voice input disabled.")
            return

        recognizer = None
        try:
            recognizer = VoiceCommandRecognizer()
            recognizer.initialize()
            recognizer.PROMPT_BIAS = (
                "Where is my bottle? Where is my phone? Where is my laptop? Where is my book? Where is my TV? What is in front of me?"
            )
            logger.info("VoiceCommandRecognizer initialized for live assistive input.")
        except Exception as rec_err:
            logger.warning(f"Could not initialize VoiceCommandRecognizer: {rec_err}")
            return

        cam_src = str(self.config.camera_index)
        audio_url = None
        if cam_src.startswith(("http://", "https://")):
            import urllib.parse
            parsed = urllib.parse.urlparse(cam_src)
            audio_url = f"{parsed.scheme}://{parsed.netloc}/audio.wav"

        def on_speech_utterance(audio_samples: np.ndarray, sample_rate: int) -> None:
            if not self._running or self._stop_event.is_set():
                return

            try:
                _, text, lat = recognizer.recognize(audio_samples, sample_rate=sample_rate)
                q_text = text.strip() if text else ""
                if not q_text or len(q_text) < 2:
                    return

                q_lower = q_text.lower()
                # 1. Universal Stop / Silence command (bypasses acoustic shield and wake word)
                is_stop = any(w in q_lower for w in ("stop", "chup", "be quiet", "stop talking", "shut up", "chup raho", "halt", "ruko", "ruk jao", "silence"))
                if is_stop:
                    logger.info(f"[USER INTERRUPT] Received STOP command: '{q_text}'. Immediately halting speech.")
                    if self.voice_manager:
                        self.voice_manager.stop()
                    if hasattr(self, "daily_service") and hasattr(self.daily_service, "db"):
                        try:
                            for act in self.daily_service.db.list_active()[:5]:
                                self.daily_service.db.mark_announced(act.id, re_alert_delay_sec=1800.0)
                        except Exception:
                            pass
                    return

                # 2. If NAYRA TTS is currently speaking and user didn't say stop, drop to prevent echo
                if self.voice_manager:
                    if hasattr(self.voice_manager, "is_speaking") and self.voice_manager.is_speaking():
                        logger.debug("Dropping speech utterance: NAYRA TTS is currently speaking.")
                        return
                    if hasattr(self.voice_manager, "shield") and self.voice_manager.shield.is_shield_active():
                        logger.debug("Dropping speech utterance: Acoustic shield cooldown active.")
                        return

                # 3. Filter Whisper hallucinations on ambient noise
                if q_lower in ("thank you.", "thank you", "thanks", "you", ".", "bye", "subscribe", "subtitles by", "watching"):
                    return

                logger.info(f"Heard speech ({lat:.1f} ms): '{q_text}'")
                self._latest_user_speech = q_text
                self._latest_speech_time = time.time()
                resp = self.process_command(q_text)
                chosen_reply = resp.spoken_response_en if self.config.default_language == "en" else (resp.spoken_response_hi or resp.spoken_response_en)
                self._latest_assistant_reply = chosen_reply or ""
                self._latest_speech_time = time.time()

                if resp.intent == UserIntent.SILENT_IGNORE:
                    logger.info(f"[Ambient Room Audio - Ignored (No Wake Word)]: '{q_text}'")
                else:
                    logger.info(f"[NAYRA Responding to '{q_text}']: {chosen_reply}")
                    print(f"\n{'='*60}\n[USER QUESTION]: {q_text}\n[NAYRA ANSWER]: {chosen_reply}\n{'='*60}\n", flush=True)
            except Exception as proc_err:
                logger.error(f"Error processing speech utterance: {proc_err}")

        receiver = None
        use_phone_mic = getattr(self.config, "use_phone_mic", False)
        if use_phone_mic and audio_url:
            import urllib.request
            try:
                probe_req = urllib.request.Request(audio_url, headers={"User-Agent": "NAYRA/1.0"})
                with urllib.request.urlopen(probe_req, timeout=1.5) as probe_resp:
                    if probe_resp.status == 200:
                        logger.info(f"Connecting PhoneAudioReceiver to: {audio_url}")
                        receiver = PhoneAudioReceiver(
                            stream_url=audio_url,
                            on_speech_utterance=on_speech_utterance,
                            energy_threshold=0.015,
                            silence_timeout_sec=0.9,
                        )
                        if receiver.start():
                            self._audio_receiver = receiver
                            logger.info("PhoneAudioReceiver active and listening to phone microphone.")
            except Exception as rx_err:
                logger.warning(f"PhoneAudioReceiver failed: {rx_err}. Falling back to laptop microphone.")
                receiver = None

        # By default, use local laptop microphone (sounddevice) for direct spoken queries
        if receiver is None:
            try:
                import sounddevice as sd
                logger.info("🎙️ Starting local laptop microphone listener via sounddevice...")
                print("\n" + "="*60 + "\n🎙️  LAPTOP MICROPHONE ACTIVE — Speak your question naturally!\n" + "="*60 + "\n", flush=True)
                audio_queue = []
                q_lock = threading.Lock()

                def sd_audio_callback(indata, frames, time_info, status):
                    if not self._running:
                        return
                    mono = indata[:, 0].copy()
                    with q_lock:
                        audio_queue.append(mono)

                with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600, callback=sd_audio_callback):
                    speech_chunks = []
                    silence_frames = 0
                    speaking = False

                    while not self._stop_event.is_set() and self._running:
                        chunk = None
                        with q_lock:
                            if audio_queue:
                                chunk = audio_queue.pop(0)
                        if chunk is None:
                            time.sleep(0.02)
                            continue

                        energy = float(np.sqrt(np.mean(chunk ** 2)))
                        # Sensitive voice activity threshold
                        if energy > 0.012:
                            speaking = True
                            silence_frames = 0
                            speech_chunks.append(chunk)
                        elif speaking:
                            speech_chunks.append(chunk)
                            silence_frames += 1
                            if silence_frames > 8:  # ~0.8s pause commits utterance
                                if len(speech_chunks) > 4:  # ~0.4s minimum speech
                                    full_audio = np.concatenate(speech_chunks)
                                    on_speech_utterance(full_audio, 16000)
                                speech_chunks = []
                                silence_frames = 0
                                speaking = False
            except Exception as mic_err:
                logger.warning(f"Local microphone listening ended: {mic_err}")
                while not self._stop_event.is_set() and self._running:
                    time.sleep(0.5)

    def shutdown(self) -> None:
        """Cleanly shutdown all subsystems."""
        if not self._running:
            return
        logger.info("Shutting down NAYRA Application...")
        self._stop_event.set()
        if self._audio_receiver and hasattr(self._audio_receiver, "stop"):
            try:
                self._audio_receiver.stop()
            except Exception:
                pass
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.5)
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        if not self.config.headless:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.orchestrator.shutdown_all()
        self.voice_manager.shutdown()
        self._running = False
        self._initialized = False
        logger.info("NAYRA Application shutdown complete.")
