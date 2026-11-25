# NAYRA — Data Security & Persistent Store Inventory

**Audit Version:** Phase 5 Product Integrity Audit  
**Date:** November 24, 2026  
**Auditor Roles:** Senior AI Product Reliability Engineer, Safety-Critical Software Reviewer, Release QA Architect  
**Principle:** 100% Local-First, Zero Cloud Leakage, Accurate Cryptographic Accounting  

---

## 1. Storage Inventory & Cryptographic Reality

The repository's persistent storage under `data/` has been systematically inspected. The table below documents every persistent store, its data classification, cryptographic status, and lifecycle controls.

| Persistent Store | Filesystem Path | Data Categories Contained | Encryption Implemented? | Encryption Mechanism | Access Controls | Retention Policy | Deletion & Backup Behavior |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **Face Recognition Vectors** | `data/face_db.bin` | Facial feature embeddings (512-d float vectors), familiar person names, relationship tags | **YES** | Authenticated symmetric encryption (`cryptography.fernet.Fernet` AES-128-CBC + HMAC-SHA256) | File key in `data/face_db.key` or `NAYRA_ENCRYPTION_KEY` env var | Persistent until user deletion | Deleted atomically upon profile wipe; encrypted binary on disk |
| **Face Database Encryption Key** | `data/face_db.key` | Cryptographic secret key used to decrypt `face_db.bin` | **NO (Plain Key)** | Plaintext base64 key file | Host OS filesystem user permissions | Persistent with face database | If deleted, encrypted face database cannot be decrypted |
| **Object Location Memory** | `data/memory/object_memory.db` | Historical object observations, COCO class labels, bounding box coordinates, spatial zones, room anchors, observation timestamps | **NO** | Standard unencrypted SQLite3 database file | Host OS filesystem user permissions | Persistent across sessions | Retained locally; updated on new detections; clearable via DB purge |
| **Caregiver / Patient Context** | `data/person_context.db` | Person sighting history, relationship metadata, interaction timestamps | **NO** | Standard unencrypted SQLite3 database file | Host OS filesystem user permissions | Persistent across sessions | Retained locally; clearable via DB purge |
| **Reminders & Medication DB** | `data/reminders.db` | Scheduled medication names, dosages/times, hydration timers, daily routine titles, completion flags, timestamps | **NO** | Standard unencrypted SQLite3 database (WAL mode) | Host OS filesystem user permissions | Persistent until completed or dismissed | Corrupted headers auto-backed up to `.corrupt.<timestamp>.bak`; user notified verbally |
| **Patient Profile & Caregivers** | `data/patient_profile.json` | Patient name, home address label, primary caregiver name, phone number, family member directory | **NO** | Plaintext unencrypted JSON file | Host OS filesystem user permissions | Configurable profile; persistent | Editable via profile manager; wiped on user reset |
| **Emergency Contacts** | `data/emergency_contacts.json` | Designated emergency contact names, phone numbers, relationship strings, primary contact flags | **NO** | Plaintext unencrypted JSON file | Host OS filesystem user permissions | Configurable contact directory | Editable via SOS configuration; wiped on user reset |
| **Emergency SOS Event Logs** | `data/emergency_alerts.jsonl` | Timestamped records of triggered SOS incidents, trigger method (voice/timeout), notification payload | **NO** | Plaintext append-only JSONL log file | Host OS filesystem user permissions | Audit log; append-only | Logged locally; never transmitted to cloud |
| **Diagnostic & Camera Frames** | `data/diagnostic_frames/`, `data/diagnostic_physical_frame.jpg` | Raw RGB camera frame captures saved during diagnostic sessions | **NO** | Standard uncompressed/compressed JPEG image files | Host OS filesystem user permissions | Diagnostic session storage | Temporary diagnostic frames; can be pruned manually |
| **OCR Diagnostic Images** | `data/ocr_diagnostics/` | Captured text images and annotated bounding box visualizations | **NO** | Standard JPEG image files | Host OS filesystem user permissions | Diagnostic session storage | Temporary diagnostic frames; can be pruned manually |
| **Synthesized Audio Samples** | `data/tts_samples/` | Synthesized test WAV audio files and benchmarking reports | **NO** | Standard 16-bit 24kHz WAV audio files | Host OS filesystem user permissions | Ephemeral test artifacts | Prunable test data |
| **Workflow Validation Results** | `data/workflow_validation_results.json` | Automated test execution metrics, intent classifications, dispatch latencies | **NO** | Plaintext JSON report | Host OS filesystem user permissions | CI/CD and verification output | Overwritten on validation runs |

---

## 2. Accurate Cryptographic Accounting & Exposure Analysis

### 2.1 Cryptographic Fact-Check
- **True Encryption:** Only `data/face_db.bin` implements genuine authenticated cryptographic encryption via `Fernet` (AES-128-CBC + HMAC-SHA256).
- **Unencrypted Stores:** SQLite databases (`reminders.db`, `object_memory.db`, `person_context.db`), JSON profile files (`patient_profile.json`, `emergency_contacts.json`), and log files (`emergency_alerts.jsonl`) are **NOT encrypted**. They reside in standard plaintext formats on the local host filesystem.
- **Documentation Correction:** Prior documentation references describing local SQLite databases as "encrypted" were inaccurate. All documentation has been updated to reflect truthful storage mechanisms.

### 2.2 Exposure Assessment & Threat Model
1. **Local Physical / Host Exposure:** An attacker with physical access or malware executing under the same user account on the host laptop can read plaintext SQLite databases and JSON configuration files.
2. **Network Exposure (Zero Threat):** Because NAYRA initiates **zero outbound network connections** (verified via `TestZeroNetworkLeakage`), no stored data is exposed across the internet or transmitted to remote cloud endpoints.
3. **Local Wi-Fi Streaming:** Live camera frames streamed from Android IP Webcam (`http://<ip>:8080/video`) travel over the local Wi-Fi LAN without TLS encryption. They are processed in memory and never persisted to continuous disk storage.

---

## 3. Safe Minimal Remediation Plan (No Cloud, No Risky Migration)

To enhance security without risking user data loss or adding cloud complexity:
1. **Host-Level OS Protection (Recommended Primary Defense):**
   - Enable Windows BitLocker Drive Encryption on the host laptop drive. This provides transparent, hardware-backed AES-256 encryption-at-rest for all files under `data/` without modifying database drivers or application logic.
   - Restrict local folder permissions on `data/` to the executing Windows user account via NTFS Access Control Lists (ACLs).
2. **Optional Database-At-Rest Encryption (Future Phase):**
   - For SQLite stores, evaluate `SQLCipher` or transparent Python-level encryption wrappers if mobile/wearable deployment requires hardware-independent database encryption.
   - **Do NOT perform ad-hoc migrations now**: In-place encryption migrations on existing user databases carry a significant risk of corruption and data loss.
3. **Diagnostic Image Pruning:**
   - Add an automated retention pruner for `data/diagnostic_frames/` so diagnostic camera snapshots older than 7 days are automatically purged.
