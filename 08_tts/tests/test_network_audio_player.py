"""Unit tests for NAYRA Module 08 NetworkAudioPlayer."""

import base64
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import time
import urllib.request

import numpy as np
import pytest

# Ensure 08_tts and root are on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
tts_dir = root_dir / "08_tts"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(tts_dir) not in sys.path:
    sys.path.insert(0, str(tts_dir))

from audio.network_audio_player import (
    HEADER_SIZE,
    MAGIC_BYTES,
    WS_GUID,
    NetworkAudioPlayer,
    PlayerState,
)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_network_player_http_endpoints():
    port = find_free_port()
    player = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)
    try:
        # 1. Test GET /
        url = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "NAYRA Audio Output" in content
            assert "Tap to Enable Phone Speaker" in content
            assert "createBufferSource" in content

        # 2. Test GET /status
        status_url = f"http://127.0.0.1:{port}/status"
        with urllib.request.urlopen(status_url, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["connected_clients"] == 0
            assert data["audio_ready_clients"] == 0
    finally:
        player.shutdown()


def test_network_player_fallback_to_laptop_when_no_phone():
    port = find_free_port()
    player = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True, fallback_to_laptop=True)
    try:
        assert not player.has_active_phone_client()
        done_called = False

        def _on_done():
            nonlocal done_called
            done_called = True

        dummy_samples = np.zeros(2400, dtype=np.float32)  # 0.1s
        player.play(dummy_samples, sample_rate=24000, on_done=_on_done)

        # In mock mode, mock thread finishes in ~0.1s
        time.sleep(0.3)
        assert done_called, "Fallback player should have completed playback"
    finally:
        player.shutdown()


def test_network_player_websocket_client_lifecycle():
    port = find_free_port()
    player = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_sock.connect(("127.0.0.1", port))
        
        # 1. Perform WebSocket handshake
        sec_key = base64.b64encode(b"0123456789abcdef").decode()
        req = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {sec_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        client_sock.sendall(req.encode("utf-8"))

        resp = client_sock.recv(1024).decode("utf-8")
        assert "101 Switching Protocols" in resp
        expected_accept = base64.b64encode(hashlib.sha1((sec_key + WS_GUID).encode("utf-8")).digest()).decode("utf-8")
        assert expected_accept in resp

        time.sleep(0.1)
        assert len(player.get_active_clients()) == 1
        assert not player.has_active_phone_client()

        # 2. Send "audio_ready" from client
        msg = json.dumps({"type": "audio_ready", "hardware_sample_rate": 48000}).encode("utf-8")
        # Client frame must be masked
        mask = b"\x11\x22\x33\x44"
        masked = bytearray(len(msg))
        for i in range(len(msg)):
            masked[i] = msg[i] ^ mask[i % 4]
        frame = bytes([0x81, 0x80 | len(msg)]) + mask + bytes(masked)
        client_sock.sendall(frame)

        time.sleep(0.2)
        assert player.has_active_phone_client()

        # 3. Test play() routes to phone client
        dummy_audio = np.linspace(-0.5, 0.5, 1200, dtype=np.float32)
        player.play(dummy_audio, sample_rate=24000)

        # Helper to read next WebSocket frame from client_sock
        def read_next_frame():
            hdr = client_sock.recv(2)
            op = hdr[0] & 0x0F
            l = hdr[1] & 0x7F
            if l == 126:
                l = struct.unpack("!H", client_sock.recv(2))[0]
            elif l == 127:
                l = struct.unpack("!Q", client_sock.recv(8))[0]
            buf = bytearray()
            while len(buf) < l:
                c = client_sock.recv(l - len(buf))
                if not c:
                    break
                buf.extend(c)
            return op, bytes(buf)

        # First frame is stop preemption (opcode 1)
        opcode, payload = read_next_frame()
        if opcode == 0x1:
            # Stop command
            stop_data = json.loads(payload.decode("utf-8"))
            assert stop_data["type"] == "stop"
            # Next frame is binary audio (opcode 2)
            opcode, payload = read_next_frame()

        assert opcode == 0x2  # Binary frame

        # Verify NAYR packet header
        assert len(payload) >= HEADER_SIZE
        magic = payload[:4]
        assert magic == MAGIC_BYTES
        version, fmt, channels, flags, utt_id, chunk_idx, sr, n_samples = struct.unpack(
            "!BBBBI I I I", payload[4:HEADER_SIZE]
        )
        assert version == 1
        assert fmt == 1  # Float32
        assert sr == 24000
        assert n_samples == 1200

        # 4. Test stop() broadcasts stop message
        player.stop()
        stop_hdr = client_sock.recv(2)
        assert stop_hdr[0] == 0x81  # Text frame
        stop_len = stop_hdr[1] & 0x7F
        stop_payload = client_sock.recv(stop_len).decode("utf-8")
        stop_json = json.loads(stop_payload)
        assert stop_json["type"] == "stop"

    finally:
        client_sock.close()
        player.shutdown()


def test_tts_manager_selecting_network_audio_player():
    port = find_free_port()
    from tts_manager import TTSManager
    tts = TTSManager(phone_speaker=True, phone_audio_port=port, mock_audio=True)
    try:
        assert isinstance(tts.player, NetworkAudioPlayer)
        assert tts.player.port == port
        # Verify speak can enqueue without error
        req_id = tts.speak("Test speech request", priority=0)
        assert req_id is not None
    finally:
        tts.shutdown()


def test_network_player_no_duplicate_server():
    port = find_free_port()
    player1 = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)
    try:
        assert player1._server is not None
        # Attempt to start second player on same port
        player2 = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)
        try:
            assert player2._server_already_running is True
            assert player2._server is None
        finally:
            player2.shutdown()
    finally:
        player1.shutdown()


def test_network_player_disconnect_during_playback():
    port = find_free_port()
    player = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_sock.connect(("127.0.0.1", port))
        sec_key = base64.b64encode(b"testdisconnectkey").decode()
        req = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {sec_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        client_sock.sendall(req.encode("utf-8"))
        client_sock.recv(1024)

        # Mark audio ready
        msg = json.dumps({"type": "audio_ready", "hardware_sample_rate": 48000}).encode("utf-8")
        mask = b"\x11\x22\x33\x44"
        masked = bytearray(len(msg))
        for i in range(len(msg)):
            masked[i] = msg[i] ^ mask[i % 4]
        frame = bytes([0x81, 0x80 | len(msg)]) + mask + bytes(masked)
        client_sock.sendall(frame)
        time.sleep(0.15)
        assert player.has_active_phone_client()

        # Start playback
        done_called = False
        def _on_done():
            nonlocal done_called
            done_called = True

        dummy = np.zeros(2400, dtype=np.float32)  # 0.1s
        player.play(dummy, sample_rate=24000, on_done=_on_done)
        assert player.is_playing()

        # Abruptly close client socket mid-playback
        client_sock.close()

        # Player should timeout or finish gracefully without hanging
        time.sleep(1.8)
        assert not player.is_playing()
        assert done_called
    finally:
        player.shutdown()

