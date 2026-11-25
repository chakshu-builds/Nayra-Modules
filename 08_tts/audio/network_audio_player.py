"""NAYRA Module 08 — Network Audio Player for Android Phone Speaker Output.

Streams Kokoro-82M synthesized audio over local Wi-Fi to an Android device
running a lightweight, zero-install Web Audio receiver in Android Chrome.

Features:
1. Pure standard library implementation (RFC 6455 WebSocket + HTTP server; no external dependencies).
2. Duck-type drop-in replacement for AudioPlayer (implements play, stop, pause, resume, is_playing, shutdown).
3. Automatic fallback to local laptop PortAudio speakers when phone is disconnected.
4. Immediate preemption (<10ms target) and utterance-ID validation to prevent stale chunks.
5. Bidirectional control: STOP triggered on phone halts laptop TTS, and vice versa.
"""

from __future__ import annotations

import base64
from enum import Enum
import hashlib
import http.server
import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional, Set

import numpy as np

try:
    from .audio_player import AudioPlayer, PlayerState
except (ImportError, ValueError):
    from audio.audio_player import AudioPlayer, PlayerState

logger = logging.getLogger("NAYRA-NetworkAudioPlayer")

# RFC 6455 Magic GUID for WebSocket handshake
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Binary audio packet header format (24 bytes):
# magic: 4s ("NAYR")
# version: B (1)
# format: B (1 = float32, 2 = int16)
# channels: B (1)
# flags: B (0)
# utterance_id: I (uint32)
# chunk_index: I (uint32)
# sample_rate: I (uint32)
# num_samples: I (uint32)
HEADER_FORMAT = "!4sBBBBI I I I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes
MAGIC_BYTES = b"NAYR"


RECEIVER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>NAYRA — Phone Speaker Receiver</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: #0b1120; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.25rem; }
        .card { background: #1e293b; border-radius: 1.25rem; border: 1px solid #334155; padding: 1.75rem; width: 100%; max-width: 440px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); text-align: center; }
        h1 { font-size: 1.6rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.25rem; letter-spacing: -0.025em; }
        p.subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 1.5rem; }
        
        .badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1.25rem; }
        .badge-disconnected { background: #450a0a; color: #fca5a5; border: 1px solid #991b1b; }
        .badge-connected { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
        .badge-locked { background: #713f12; color: #fde047; border: 1px solid #ca8a04; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .dot-red { background: #ef4444; }
        .dot-green { background: #10b981; box-shadow: 0 0 10px #10b981; }
        .dot-yellow { background: #eab308; }

        .btn { width: 100%; padding: 1rem 1.25rem; border: none; border-radius: 0.875rem; font-size: 1.05rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem; }
        .btn-enable { background: #0284c7; color: white; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4); }
        .btn-enable:hover { background: #0369a1; }
        .btn-enable:active { transform: scale(0.98); }
        .btn-enabled { background: #059669; color: white; cursor: default; }

        .btn-stop { background: #dc2626; color: white; display: none; }
        .btn-stop:hover { background: #b91c1c; }
        .btn-stop:active { transform: scale(0.98); }

        .btn-test { background: #334155; color: #e2e8f0; font-size: 0.875rem; padding: 0.6rem; border-radius: 0.6rem; }
        .btn-test:hover { background: #475569; }

        .stats-box { background: #0f172a; border-radius: 0.75rem; border: 1px solid #1e293b; padding: 0.85rem; font-size: 0.8rem; color: #94a3b8; text-align: left; margin-top: 1rem; line-height: 1.5; font-family: monospace; height: 130px; overflow-y: auto; }
        .footer { margin-top: 1.25rem; font-size: 0.75rem; color: #64748b; }
    </style>
</head>
<body>
    <div class="card">
        <h1>NAYRA Audio Output</h1>
        <p class="subtitle">Android Phone Speaker Output Sink</p>

        <div id="netStatus" class="badge badge-disconnected">
            <span class="dot dot-red"></span> <span id="netText">Disconnected</span>
        </div>

        <button id="btnEnable" class="btn btn-enable" onclick="enableAudioEngine()">
            🔊 Tap to Enable Phone Speaker
        </button>

        <button id="btnStop" class="btn btn-stop" onclick="userStopSpeech()">
            ⏹ Stop Reading
        </button>

        <button id="btnTestChime" class="btn btn-test" onclick="playTestChime()" style="display:none;">
            🔔 Play Test Chime
        </button>

        <div id="logBox" class="stats-box">
            [SYS] Waiting for user tap to initialize Web Audio...
        </div>

        <p class="footer">NAYRA Offline Assistive AI Subsystem</p>
    </div>

    <script>
        let ws = null;
        let audioCtx = null;
        let audioEnabled = false;
        let activeSources = [];
        let nextStartTime = 0;
        let currentUtteranceId = null;

        const netStatus = document.getElementById('netStatus');
        const netText = document.getElementById('netText');
        const btnEnable = document.getElementById('btnEnable');
        const btnStop = document.getElementById('btnStop');
        const btnTestChime = document.getElementById('btnTestChime');
        const logBox = document.getElementById('logBox');

        function appendLog(msg) {
            const time = new Date().toLocaleTimeString();
            logBox.innerHTML += `<div>[${time}] ${msg}</div>`;
            logBox.scrollTop = logBox.scrollHeight;
        }

        async function enableAudioEngine() {
            try {
                if (!audioCtx) {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }
                if (audioCtx.state === 'suspended') {
                    await audioCtx.resume();
                }

                if (audioCtx.state === 'running') {
                    audioEnabled = true;
                    btnEnable.textContent = "✓ Phone Speaker Enabled";
                    btnEnable.className = "btn btn-enabled";
                    btnEnable.onclick = null;
                    btnStop.style.display = "flex";
                    btnTestChime.style.display = "block";
                    appendLog("Web Audio API initialized (Hardware SR: " + audioCtx.sampleRate + " Hz)");
                    playTestChime();

                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: "audio_ready",
                            hardware_sample_rate: audioCtx.sampleRate
                        }));
                    }
                } else {
                    appendLog("Audio state suspended; please tap again.");
                }
            } catch (err) {
                appendLog("Error enabling audio: " + err.message);
            }
        }

        function playTestChime() {
            if (!audioCtx || audioCtx.state !== 'running') return;
            try {
                const now = audioCtx.currentTime;
                // Dual chime 523Hz (C5) -> 659Hz (E5)
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = "sine";
                osc.frequency.setValueAtTime(523.25, now);
                osc.frequency.setValueAtTime(659.25, now + 0.09);
                gain.gain.setValueAtTime(0.08, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now);
                osc.stop(now + 0.28);
                appendLog("Played test chime through phone speaker.");
            } catch (err) {
                appendLog("Chime error: " + err);
            }
        }

        function userStopSpeech() {
            stopAllAudio();
            appendLog("User tapped STOP reading.");
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "user_stop" }));
            }
        }

        function stopAllAudio() {
            for (const src of activeSources) {
                try {
                    src.stop();
                    src.disconnect();
                } catch (e) {}
            }
            activeSources = [];
            currentUtteranceId = null;
            if (audioCtx) {
                nextStartTime = audioCtx.currentTime;
            }
        }

        function connectWebSocket() {
            const loc = window.location;
            const wsUri = (loc.protocol === "https:" ? "wss://" : "ws://") + loc.host + "/ws";
            appendLog("Connecting to NAYRA audio server at " + wsUri + "...");

            ws = new WebSocket(wsUri);
            ws.binaryType = "arraybuffer";

            ws.onopen = () => {
                netStatus.className = "badge badge-connected";
                netStatus.innerHTML = '<span class="dot dot-green"></span> <span id="netText">Connected (Phone Speaker Active)</span>';
                appendLog("Connected to laptop audio transmitter.");

                if (audioEnabled && audioCtx && audioCtx.state === 'running') {
                    ws.send(JSON.stringify({
                        type: "audio_ready",
                        hardware_sample_rate: audioCtx.sampleRate
                    }));
                }
            };

            ws.onclose = () => {
                netStatus.className = "badge badge-disconnected";
                netStatus.innerHTML = '<span class="dot dot-red"></span> <span id="netText">Disconnected (Reconnecting...)</span>';
                appendLog("Connection lost. Retrying in 1.5s...");
                stopAllAudio();
                setTimeout(connectWebSocket, 1500);
            };

            ws.onerror = (err) => {
                appendLog("WebSocket error detected.");
            };

            ws.onmessage = (event) => {
                if (typeof event.data === "string") {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === "stop") {
                            stopAllAudio();
                            appendLog("STOP command received for utterance #" + (msg.utterance_id || "all"));
                        } else if (msg.type === "pause") {
                            if (audioCtx && audioCtx.state === 'running') {
                                audioCtx.suspend();
                                appendLog("Playback paused.");
                            }
                        } else if (msg.type === "resume") {
                            if (audioCtx && audioCtx.state === 'suspended') {
                                audioCtx.resume();
                                appendLog("Playback resumed.");
                            }
                        } else if (msg.type === "test_tone") {
                            playTestChime();
                        }
                    } catch (e) {
                        appendLog("Invalid JSON: " + event.data);
                    }
                } else if (event.data instanceof ArrayBuffer) {
                    handleAudioPacket(event.data);
                }
            };
        }

        function handleAudioPacket(buffer) {
            if (!audioEnabled || !audioCtx || audioCtx.state !== 'running') {
                appendLog("Audio packet dropped: Tap button to enable audio first.");
                return;
            }
            if (buffer.byteLength < 24) return;

            const view = new DataView(buffer);
            // Verify magic "NAYR"
            const m0 = view.getUint8(0), m1 = view.getUint8(1), m2 = view.getUint8(2), m3 = view.getUint8(3);
            if (m0 !== 0x4E || m1 !== 0x41 || m2 !== 0x59 || m3 !== 0x52) {
                appendLog("Invalid packet magic header.");
                return;
            }

            const format = view.getUint8(5); // 1 = float32, 2 = int16
            const utteranceId = view.getUint32(8, false);
            const chunkIndex = view.getUint32(12, false);
            const sampleRate = view.getUint32(16, false);
            const numSamples = view.getUint32(20, false);

            if (currentUtteranceId !== utteranceId) {
                stopAllAudio();
                currentUtteranceId = utteranceId;
                nextStartTime = audioCtx.currentTime;
                appendLog("Started speech utterance #" + utteranceId);
            }

            // Create Web Audio buffer with Kokoro's native 24 kHz sample rate
            const audioBuffer = audioCtx.createBuffer(1, numSamples, sampleRate);
            const channelData = audioBuffer.getChannelData(0);

            if (format === 1) { // Float32
                const floatView = new Float32Array(buffer, 24, numSamples);
                channelData.set(floatView);
            } else if (format === 2) { // Int16
                const intView = new Int16Array(buffer, 24, numSamples);
                for (let i = 0; i < numSamples; i++) {
                    channelData[i] = intView[i] / 32768.0;
                }
            }

            const source = audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioCtx.destination);

            const now = audioCtx.currentTime;
            if (nextStartTime < now) {
                nextStartTime = now + 0.005; // 5ms lead time prevents underrun clicks
            }

            source.start(nextStartTime);
            nextStartTime += audioBuffer.duration;
            activeSources.push(source);

            source.onended = () => {
                const idx = activeSources.indexOf(source);
                if (idx !== -1) activeSources.splice(idx, 1);
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: "chunk_done",
                        utterance_id: utteranceId,
                        chunk_index: chunkIndex
                    }));
                }
            };
        }

        // Start connection
        connectWebSocket();
    </script>
</body>
</html>
"""


class WebSocketClient:
    """Encapsulates a connected WebSocket client socket with thread-safe send."""

    def __init__(self, sock: socket.socket, addr: tuple[str, int]) -> None:
        self.sock = sock
        self.addr = addr
        self.audio_ready = False
        self.hardware_sample_rate = 48000
        self._send_lock = threading.Lock()
        self.is_alive = True

    def send_text(self, text: str) -> bool:
        """Send a UTF-8 text message frame (opcode 0x1)."""
        payload = text.encode("utf-8")
        frame = self._build_frame(payload, opcode=0x1)
        return self._send_bytes(frame)

    def send_binary(self, data: bytes) -> bool:
        """Send a binary message frame (opcode 0x2)."""
        frame = self._build_frame(data, opcode=0x2)
        return self._send_bytes(frame)

    def send_ping(self) -> bool:
        """Send a ping frame (opcode 0x9)."""
        frame = self._build_frame(b"", opcode=0x9)
        return self._send_bytes(frame)

    def _build_frame(self, payload: bytes, opcode: int) -> bytes:
        fin_and_opcode = 0x80 | (opcode & 0x0F)
        length = len(payload)
        if length <= 125:
            header = bytes([fin_and_opcode, length])
        elif length <= 65535:
            header = struct.pack("!BBH", fin_and_opcode, 126, length)
        else:
            header = struct.pack("!BBQ", fin_and_opcode, 127, length)
        return header + payload

    def _send_bytes(self, data: bytes) -> bool:
        if not self.is_alive:
            return False
        with self._send_lock:
            try:
                self.sock.sendall(data)
                return True
            except Exception as exc:
                logger.debug("Error sending to %s: %s", self.addr, exc)
                self.is_alive = False
                return False

    def close(self) -> None:
        """Safely close client socket."""
        self.is_alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class NetworkAudioPlayer:
    """Network-streaming Audio Player targeting Android Phone Speaker.

    Conforms fully to AudioPlayer interface:
    - play(samples, sample_rate, on_done)
    - stop()
    - pause()
    - resume()
    - is_playing()
    - get_state()
    - shutdown()

    Routing Policy:
    1. If an active Android phone browser client has tapped 'Enable Phone Speaker',
       audio is streamed as 24kHz float32 PCM over local Wi-Fi WebSocket directly to the phone.
    2. If no phone client is connected or ready, logs a warning and routes to the local
       laptop PortAudio player fallback.
    """

    DEFAULT_SAMPLE_RATE: int = 24000

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8088,
        mock_mode: bool = False,
        fallback_to_laptop: bool = True,
        on_user_stop_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.mock_mode = mock_mode
        self.fallback_to_laptop = fallback_to_laptop
        self.on_user_stop_callback = on_user_stop_callback

        self._state = PlayerState.IDLE
        self._lock = threading.RLock()

        # Utterance tracking to prevent stale chunks
        self._current_utterance_id = 0
        self._current_chunk_index = 0

        # Active playback tracking
        self._active_on_done: Optional[Callable[[], None]] = None
        self._chunk_done_event = threading.Event()
        self._playing_on_phone = False
        self._server_already_running = False

        # Connected WebSocket clients
        self._clients_lock = threading.Lock()
        self._clients: Set[WebSocketClient] = set()

        # Fallback local PortAudio player
        self._laptop_player = AudioPlayer(
            sample_rate=self.DEFAULT_SAMPLE_RATE,
            mock_mode=mock_mode,
        )

        # HTTP/WebSocket server thread
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._stop_server_event = threading.Event()

        self._start_server()

    def _start_server(self) -> None:
        """Bind and start the background HTTP/WebSocket server."""
        # Pre-flight check: ensure no duplicate server is already listening
        check_host = "127.0.0.1" if self.host in ("0.0.0.0", "", "127.0.0.1") else self.host
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.3)
                if probe.connect_ex((check_host, self.port)) == 0:
                    logger.warning(
                        "Port %d is already in use by an active receiver server. "
                        "Reusing existing endpoint without starting a duplicate server.",
                        self.port,
                    )
                    self._server = None
                    self._server_already_running = True
                    return
        except Exception:
            pass

        player_ref = self

        class CombinedHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                logger.debug("%s - " + format, self.address_string(), *args)

            def do_GET(self):
                logger.info("Incoming HTTP request: GET %s from %s", self.path, self.client_address)
                if self.path == "/" or self.path.startswith("/index"):
                    body = RECEIVER_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    active_cnt = sum(1 for c in player_ref.get_active_clients() if c.audio_ready)
                    data = json.dumps({
                        "connected_clients": len(player_ref._clients),
                        "audio_ready_clients": active_cnt,
                        "utterance_id": player_ref._current_utterance_id,
                        "state": player_ref._state.value,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/test_chime":
                    player_ref.broadcast_test_chime()
                    resp = b'{"status": "chime_sent"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                elif self.path == "/ws":
                    self._handle_websocket_upgrade()
                else:
                    self.send_error(404, "Not Found")

            def _handle_websocket_upgrade(self):
                key = self.headers.get("Sec-WebSocket-Key")
                if not key:
                    self.send_error(400, "Missing Sec-WebSocket-Key")
                    return

                accept_key = base64.b64encode(
                    hashlib.sha1((key.strip() + WS_GUID).encode("utf-8")).digest()
                ).decode("utf-8")

                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept_key)
                self.end_headers()
                self.wfile.flush()

                # Register connected WebSocket client
                client = WebSocketClient(self.connection, self.client_address)
                player_ref._register_client(client)
                logger.info("Android Phone connected to WebSocket from %s", self.client_address)

                # Enter frame reading loop
                try:
                    self._ws_loop(client)
                finally:
                    player_ref._unregister_client(client)
                    logger.info("Android Phone disconnected from %s", self.client_address)

            def _ws_loop(self, client: WebSocketClient):
                sock = client.sock
                sock.settimeout(2.0)
                while not player_ref._stop_server_event.is_set() and client.is_alive:
                    try:
                        header = sock.recv(2)
                        if not header or len(header) < 2:
                            break
                        b1, b2 = header
                        opcode = b1 & 0x0F
                        is_masked = bool(b2 & 0x80)
                        length = b2 & 0x7F

                        if length == 126:
                            ext_len = sock.recv(2)
                            if len(ext_len) < 2:
                                break
                            length = struct.unpack("!H", ext_len)[0]
                        elif length == 127:
                            ext_len = sock.recv(8)
                            if len(ext_len) < 8:
                                break
                            length = struct.unpack("!Q", ext_len)[0]

                        mask = sock.recv(4) if is_masked else b""
                        if is_masked and len(mask) < 4:
                            break

                        payload = bytearray()
                        while len(payload) < length:
                            chunk = sock.recv(length - len(payload))
                            if not chunk:
                                break
                            payload.extend(chunk)

                        if is_masked:
                            for i in range(len(payload)):
                                payload[i] ^= mask[i % 4]

                        # Opcode processing
                        if opcode == 0x8:  # Close
                            break
                        elif opcode == 0x9:  # Ping -> Pong
                            client._send_bytes(client._build_frame(bytes(payload), opcode=0xA))
                        elif opcode == 0x1:  # Text frame (JSON)
                            try:
                                msg = json.loads(payload.decode("utf-8"))
                                player_ref._handle_client_message(client, msg)
                            except Exception as json_err:
                                logger.debug("Malformed client JSON: %s", json_err)
                    except socket.timeout:
                        continue
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        break
                    except Exception as exc:
                        logger.debug("WebSocket client error: %s", exc)
                        break

        try:
            self._server = http.server.ThreadingHTTPServer((self.host, self.port), CombinedHandler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="NAYRA-NetworkAudioServer",
                daemon=True,
            )
            self._server_thread.start()
            logger.info("NetworkAudioPlayer server listening at http://%s:%d", self.host, self.port)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 10048 or getattr(exc, "errno", None) in (10048, 98, 48) or "already in use" in str(exc).lower():
                logger.warning(
                    "Port %d is already in use by an active server. Reusing endpoint without starting duplicate.",
                    self.port,
                )
                self._server = None
                self._server_already_running = True
            else:
                logger.error("Failed to start NetworkAudioPlayer server on port %d: %s", self.port, exc)
                self._server = None
        except Exception as exc:
            logger.error("Failed to start NetworkAudioPlayer server on port %d: %s", self.port, exc)
            self._server = None

    def _register_client(self, client: WebSocketClient) -> None:
        with self._clients_lock:
            self._clients.add(client)

    def _unregister_client(self, client: WebSocketClient) -> None:
        client.close()
        with self._clients_lock:
            self._clients.discard(client)

    def get_active_clients(self) -> list[WebSocketClient]:
        """Return list of alive clients."""
        with self._clients_lock:
            return [c for c in self._clients if c.is_alive]

    def has_active_phone_client(self) -> bool:
        """Return True if at least one phone client is connected with Web Audio active."""
        with self._clients_lock:
            return any(c.is_alive and c.audio_ready for c in self._clients)

    def _handle_client_message(self, client: WebSocketClient, msg: Dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "audio_ready":
            client.audio_ready = True
            client.hardware_sample_rate = msg.get("hardware_sample_rate", 48000)
            logger.info(
                "Phone Audio Enabled! (%s, HW sample rate: %d Hz)",
                client.addr,
                client.hardware_sample_rate,
            )
        elif msg_type == "chunk_done":
            utt_id = msg.get("utterance_id")
            chunk_idx = msg.get("chunk_index")
            logger.debug("Received chunk_done ack: utt=%s, chunk=%s", utt_id, chunk_idx)
            self._chunk_done_event.set()
        elif msg_type == "user_stop":
            logger.info("STOP requested from phone screen by user.")
            self.stop()
            if self.on_user_stop_callback:
                self.on_user_stop_callback()

    def broadcast_test_chime(self) -> None:
        """Trigger test chime on all connected clients."""
        msg = json.dumps({"type": "test_tone"})
        with self._clients_lock:
            for c in self._clients:
                c.send_text(msg)

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """Queue and begin playback of audio samples.

        Routes to Android Phone Speaker if connected and active, or falls back
        to laptop PortAudio speakers.
        """
        with self._lock:
            self.stop()

            # Prepare samples
            data = np.asarray(samples, dtype=np.float32).ravel()
            if len(data) == 0:
                if on_done:
                    on_done()
                return

            self._current_utterance_id += 1
            self._current_chunk_index += 1
            current_utt = self._current_utterance_id
            current_chunk = self._current_chunk_index

            phone_available = self.has_active_phone_client()

            if phone_available:
                self._playing_on_phone = True
                self._state = PlayerState.PLAYING
                self._chunk_done_event.clear()
                self._active_on_done = on_done

                # Pack 24-byte NAYR binary header
                # format=1 (float32), channels=1
                num_samples = len(data)
                raw_audio_bytes = data.tobytes()
                header = struct.pack(
                    HEADER_FORMAT,
                    MAGIC_BYTES,
                    1,   # version
                    1,   # format (1 = float32)
                    1,   # channels
                    0,   # flags
                    current_utt,
                    current_chunk,
                    sample_rate,
                    num_samples,
                )
                packet = header + raw_audio_bytes

                # Transmit binary packet to all active phone clients
                sent_any = False
                with self._clients_lock:
                    for client in self._clients:
                        if client.is_alive and client.audio_ready:
                            if client.send_binary(packet):
                                sent_any = True

                if not sent_any:
                    logger.warning("Failed to send audio to phone; falling back to laptop.")
                    self._playing_on_phone = False
                    self._laptop_player.play(data, sample_rate, on_done)
                    return

                # Spawn background watcher for chunk completion or fallback timeout
                duration_sec = len(data) / sample_rate

                def _phone_completion_watcher():
                    # Wait for phone acknowledgment, stop signal, or timeout
                    t_start = time.monotonic()
                    # Add 1.5s tolerance to audio duration for network latency & scheduling
                    timeout = duration_sec + 1.5
                    while time.monotonic() - t_start < timeout:
                        if not self._playing_on_phone or self._state != PlayerState.PLAYING:
                            return
                        if self._chunk_done_event.is_set():
                            break
                        time.sleep(0.01)

                    with self._lock:
                        if self._playing_on_phone and current_utt == self._current_utterance_id:
                            self._state = PlayerState.IDLE
                            self._playing_on_phone = False
                            cb = self._active_on_done
                            self._active_on_done = None
                            if cb is not None:
                                threading.Thread(target=cb, daemon=True).start()

                threading.Thread(target=_phone_completion_watcher, daemon=True).start()

            else:
                if self.fallback_to_laptop:
                    logger.warning(
                        "No phone receiver ready at http://%s:%d; routing speech to laptop speakers.",
                        self.host,
                        self.port,
                    )
                    self._playing_on_phone = False
                    self._state = PlayerState.PLAYING

                    def _laptop_done():
                        with self._lock:
                            self._state = PlayerState.IDLE
                        if on_done:
                            on_done()

                    self._laptop_player.play(data, sample_rate, _laptop_done)
                else:
                    logger.warning("Phone receiver disconnected and laptop fallback disabled.")
                    if on_done:
                        on_done()

    def stop(self) -> None:
        """Immediately halt active playback, broadcast STOP to phone, and clear buffers."""
        with self._lock:
            self._state = PlayerState.STOPPED
            self._chunk_done_event.set()
            self._playing_on_phone = False
            self._active_on_done = None

            # Invalidate pending chunks by incrementing utterance ID
            self._current_utterance_id += 1

            # Stop laptop player fallback
            self._laptop_player.stop()

            # Broadcast stop to all connected phone clients
            stop_msg = json.dumps({
                "type": "stop",
                "utterance_id": self._current_utterance_id,
            })
            with self._clients_lock:
                for client in self._clients:
                    client.send_text(stop_msg)

            self._state = PlayerState.IDLE

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if self._state == PlayerState.PLAYING:
                self._state = PlayerState.PAUSED
                if self._playing_on_phone:
                    pause_msg = json.dumps({"type": "pause"})
                    with self._clients_lock:
                        for client in self._clients:
                            client.send_text(pause_msg)
                else:
                    self._laptop_player.pause()

    def resume(self) -> None:
        """Resume playback."""
        with self._lock:
            if self._state == PlayerState.PAUSED:
                self._state = PlayerState.PLAYING
                if self._playing_on_phone:
                    resume_msg = json.dumps({"type": "resume"})
                    with self._clients_lock:
                        for client in self._clients:
                            client.send_text(resume_msg)
                else:
                    self._laptop_player.resume()

    def is_playing(self) -> bool:
        """Return True if phone or laptop is actively playing audio."""
        with self._lock:
            if self._playing_on_phone:
                return self._state == PlayerState.PLAYING
            return self._laptop_player.is_playing()

    def get_state(self) -> PlayerState:
        """Return current PlayerState enum."""
        with self._lock:
            return self._state

    def shutdown(self) -> None:
        """Cleanly close server, sockets, and local player."""
        logger.info("Shutting down NetworkAudioPlayer...")
        self.stop()
        self._stop_server_event.set()

        # Disconnect all WebSocket clients
        with self._clients_lock:
            for client in list(self._clients):
                client.close()
            self._clients.clear()

        # Stop HTTP server
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as exc:
                logger.debug("Error shutting down HTTP server: %s", exc)
            self._server = None

        self._laptop_player.shutdown()
        logger.info("NetworkAudioPlayer shutdown completed.")
