# NAYRA — Privacy & Local-First Data Policy

## 1. Privacy Principles

NAYRA is built from the ground up on the principle of **Zero Cloud Leakage**. 

Assistive technologies often process deeply sensitive personal data: live video streams of personal living spaces, audio recordings of private family conversations, medical schedules, and caregiver contacts. NAYRA guarantees complete user data privacy through strict architectural isolation.

---

## 2. Local-First Processing Architecture

1. **No External Cloud API Calls**:
   - All machine learning inference (YOLOv10n object detection, ByteTrack tracking, Monocular Depth, Face Recognition, PaddleOCR / EasyOCR, Whisper STT, Kokoro TTS) runs entirely on the host device.
   - NAYRA operates without internet connectivity.

2. **No Telemetry or Tracking**:
   - NAYRA contains zero tracking SDKs, analytics beacons, or remote telemetry reporting.
   - Diagnostic logging is written exclusively to local log files on the host machine.

---

## 3. Data Storage & Cryptographic Reality

All persistent application data resides in the local `data/` directory. NAYRA adheres to truthful cryptographic reporting:

| Data Type | Storage Format | Actual Filesystem Location | Encryption Implemented? | Cryptographic Mechanism / Access Control |
| :--- | :--- | :--- | :---: | :--- |
| **Face Recognition Database** | Encrypted Binary | `data/face_db.bin` | **YES** | Authenticated Symmetric Encryption (`Fernet` AES-128-CBC + HMAC-SHA256) |
| **Face DB Encryption Key** | Plaintext Key File | `data/face_db.key` | **NO** | Base64 key file; protected via OS file permissions or `NAYRA_ENCRYPTION_KEY` |
| **Object Location Memory** | SQLite Database | `data/memory/object_memory.db` | **NO** | Standard unencrypted SQLite3 database |
| **Medication & Reminders** | SQLite Database | `data/reminders.db` | **NO** | Standard unencrypted SQLite3 database (WAL mode) |
| **Patient Profile & Caregivers**| JSON | `data/patient_profile.json` | **NO** | Standard unencrypted JSON format |
| **Emergency Contacts** | JSON | `data/emergency_contacts.json` | **NO** | Standard unencrypted JSON format |
| **Emergency SOS Event Log** | JSONL | `data/emergency_alerts.jsonl`| **NO** | Plaintext append-only JSONL log |
| **Diagnostic Snapshots** | JPEG Images | `data/diagnostic_frames/` | **NO** | Standard uncompressed/compressed JPEG image files |

> [!NOTE]
> Only face feature vectors are cryptographically encrypted via Fernet. Relational SQLite databases and JSON configuration files are stored unencrypted on the local host filesystem. Full data-at-rest protection is provided via host operating system disk encryption (e.g. Windows BitLocker). Zero cloud leakage ensures local data is never exposed across the network.

---

## 4. User Data Ownership & Deletion

Users and caregivers maintain full control over all stored data:
- **Instant Wipe**: Deleting the `data/` folder immediately and irrevocably erases all memories, reminder schedules, familiar person profiles, and emergency logs.
- **Clear Commands**: Spoken commands such as *"Clear memory"* or *"Delete reminders"* purge specific database records locally without leaving lingering backups.
- **Exportability**: All configuration and profile files use transparent standard formats (JSON and SQLite) for easy backup and inspection.
