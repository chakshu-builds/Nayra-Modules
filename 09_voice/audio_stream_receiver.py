"""NAYRA Module 09 — Phone Microphone Audio Stream Receiver.

Streams live microphone audio from Android IP Webcam (/audio.wav),
parses PCM frames, monitors voice activity (VAD), and extracts speech segments.
"""

from __future__ import annotations

import collections
import io
import logging
import struct
import threading
import time
from typing import Callable, Optional
import urllib.error
import urllib.request

import numpy as np

logger = logging.getLogger("NAYRA-PhoneAudioReceiver")


class PhoneAudioReceiver:
    """Receives and buffers live microphone audio streamed from Android IP Webcam.

    Features:
    1. Robust HTTP streaming of /audio.wav with dynamic WAV header inspection.
    2. Real-time energy-based Voice Activity Detection (VAD).
    3. Asynchronous speech utterance segmentation.
    4. Resampling to standard 16,000 Hz mono float32 for neural STT models.
    5. Mock / injection mode for offline testing without live Android device.
    """

    DEFAULT_STREAM_URL: str = "http://192.168.0.101:8080/audio.wav"

    def __init__(
        self,
        stream_url: str = DEFAULT_STREAM_URL,
        target_sample_rate: int = 16000,
        energy_threshold: float = 0.008,
        silence_timeout_sec: float = 0.8,
        min_speech_duration_sec: float = 0.35,
        max_speech_duration_sec: float = 8.0,
        on_speech_utterance: Optional[Callable[[np.ndarray, int], None]] = None,
        mock_mode: bool = False,
    ) -> None:
        self.stream_url = self._normalize_audio_url(stream_url)
        self.target_sample_rate = target_sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_speech_duration_sec = max_speech_duration_sec
        self.on_speech_utterance = on_speech_utterance
        self.mock_mode = mock_mode

        # Stream properties parsed from WAV header
        self.source_sample_rate: int = 16000
        self.source_channels: int = 1
        self.source_bits_per_sample: int = 16

        # State tracking
        self._is_running = False
        self._is_connected = False
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _normalize_audio_url(url: str) -> str:
        """Ensure IP Webcam URL points to /audio.wav."""
        u = url.strip()
        if not u.startswith(("http://", "https://")):
            u = f"http://{u}"
        if not u.endswith(".wav"):
            if u.endswith("/"):
                u = f"{u}audio.wav"
            else:
                u = f"{u}/audio.wav"
        return u

    def start(self) -> bool:
        """Start background audio streaming receiver thread."""
        if self._is_running:
            return True

        self._is_running = True
        self._thread = threading.Thread(
            target=self._stream_loop,
            name="NAYRA-PhoneAudioReceiver",
            daemon=True,
        )
        self._thread.start()
        logger.info("PhoneAudioReceiver thread started (mock_mode=%s).", self.mock_mode)
        return True

    def stop(self) -> None:
        """Stop streaming receiver."""
        self._is_running = False
        self._is_connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("PhoneAudioReceiver stopped.")

    def inject_audio_sample(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        """Manually deliver an audio sample (used in tests or offline simulation)."""
        if self.on_speech_utterance:
            # Resample if needed
            data = np.asarray(samples, dtype=np.float32).ravel()
            if sample_rate != self.target_sample_rate:
                num_out = int(len(data) * self.target_sample_rate / sample_rate)
                data = np.interp(
                    np.linspace(0, len(data), num_out, endpoint=False),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
            self.on_speech_utterance(data, self.target_sample_rate)

    def _read_exact(self, stream: io.RawIOBase, n_bytes: int) -> Optional[bytes]:
        """Read exactly n_bytes from stream, handling fragmented TCP packets."""
        buf = bytearray()
        while len(buf) < n_bytes and self._is_running:
            try:
                chunk = stream.read(n_bytes - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            except Exception as exc:
                logger.debug("Socket read error: %s", exc)
                return None
        return bytes(buf) if len(buf) == n_bytes else None

    def _stream_loop(self) -> None:
        """Dedicated streaming loop connecting to HTTP endpoint."""
        while self._is_running:
            if self.mock_mode:
                time.sleep(0.1)
                continue

            try:
                logger.info("Connecting to Android microphone stream: %s...", self.stream_url)
                req = urllib.request.Request(
                    self.stream_url,
                    headers={"User-Agent": "NAYRA-Assistive/1.0", "Accept": "audio/wav, audio/*"},
                )
                with urllib.request.urlopen(req, timeout=15.0) as response:
                    logger.info("Connected to phone audio stream (Status: %s).", response.status)
                    self._is_connected = True
                    self._last_error = None

                    # Read 44-byte WAV header
                    header = self._read_exact(response, 44)
                    if header and len(header) >= 44:
                        self._parse_wav_header(header)

                    self._process_pcm_stream(response)

            except Exception as exc:
                self._is_connected = False
                self._last_error = str(exc)
                logger.warning("Microphone stream error: %s. Retrying in 2.0s...", exc)
                time.sleep(2.0)

    def _parse_wav_header(self, header: bytes) -> None:
        """Parse standard RIFF WAVE header to extract audio specs."""
        try:
            riff, size, wave = struct.unpack("<4sI4s", header[:12])
            if riff == b"RIFF" and wave == b"WAVE":
                fmt, fmt_size, audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
                    "<4sIHHIIHH", header[12:36]
                )
                self.source_channels = channels
                self.source_sample_rate = sample_rate
                self.source_bits_per_sample = bits
                logger.info(
                    "Parsed WAV header: rate=%d Hz, channels=%d, bits=%d",
                    sample_rate,
                    channels,
                    bits,
                )
        except Exception as exc:
            logger.warning("Could not parse standard WAV header (%s); using defaults.", exc)

    def _process_pcm_stream(self, stream: io.RawIOBase) -> None:
        """Read 16-bit PCM stream, apply VAD, and emit speech utterances."""
        bytes_per_sample = 2 * self.source_channels
        chunk_samples = 1024
        chunk_bytes = chunk_samples * bytes_per_sample

        pre_roll_buffer: collections.deque = collections.deque(maxlen=4)
        speech_buffer: list[np.ndarray] = []
        is_speech_active = False
        silence_start_time = 0.0
        speech_start_time = 0.0

        while self._is_running:
            raw = self._read_exact(stream, chunk_bytes)
            if raw is None:
                break

            # Convert 16-bit PCM bytes to float32
            int_data = np.frombuffer(raw, dtype=np.int16)
            if self.source_channels > 1:
                int_data = int_data[:: self.source_channels]  # Take first channel

            float_data = (int_data.astype(np.float32) / 32768.0)

            # Resample chunk to target 16000 Hz if needed
            if self.source_sample_rate != self.target_sample_rate:
                num_out = int(len(float_data) * self.target_sample_rate / self.source_sample_rate)
                chunk_16k = np.interp(
                    np.linspace(0, len(float_data), num_out, endpoint=False),
                    np.arange(len(float_data)),
                    float_data,
                ).astype(np.float32)
            else:
                chunk_16k = float_data

            # Calculate short-time RMS energy (remove DC bias to prevent false voice activity on phone mics)
            chunk_ac = chunk_16k - np.mean(chunk_16k) if len(chunk_16k) > 0 else chunk_16k
            rms = float(np.sqrt(np.mean(chunk_ac ** 2))) if len(chunk_ac) > 0 else 0.0
            now = time.time()

            if rms >= self.energy_threshold:
                # Speech detected
                if not is_speech_active:
                    is_speech_active = True
                    speech_start_time = now
                    # Include pre-roll audio so initial consonants/syllables are preserved
                    speech_buffer = list(pre_roll_buffer)
                    logger.debug("Speech activity onset detected (RMS=%.4f, pre-roll=%d chunks).", rms, len(pre_roll_buffer))

                speech_buffer.append(chunk_16k)
                silence_start_time = 0.0

            elif is_speech_active:
                speech_buffer.append(chunk_16k)
                if silence_start_time == 0.0:
                    silence_start_time = now

                # Check if silence timeout exceeded or max duration reached
                silence_dur = now - silence_start_time
                total_speech_dur = now - speech_start_time

                if silence_dur >= self.silence_timeout_sec or total_speech_dur >= self.max_speech_duration_sec:
                    is_speech_active = False
                    if total_speech_dur >= self.min_speech_duration_sec and self._is_running:
                        utterance = np.concatenate(speech_buffer)
                        logger.info("Speech utterance captured (duration=%.2fs).", len(utterance) / self.target_sample_rate)
                        if self.on_speech_utterance and self._is_running:
                            self.on_speech_utterance(utterance, self.target_sample_rate)
                    speech_buffer = []
            else:
                pre_roll_buffer.append(chunk_16k)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
