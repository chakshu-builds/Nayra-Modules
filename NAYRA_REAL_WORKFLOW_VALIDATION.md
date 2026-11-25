# NAYRA — Real Assistive Feature Execution & Validation Matrix

**Audit Version:** Phase 5 Product Integrity Audit  
**Date:** November 24, 2026  
**Auditor Roles:** Senior AI Product Reliability Engineer, Safety-Critical Software Reviewer, Release QA Architect  
**Validation Standard:** True Service Invocation & Pipeline Trace (Beyond Intent Classification)  

---

## 1. Trace Matrix: Real Service Invocation Across Modules 10–15

This matrix verifies the complete execution pipeline from voice input, through orchestrator dispatch, to actual domain service execution and spoken voice response generation.

| Workflow & Module | User Command | Detected Intent | Dispatched Module | Actual Function Invoked | Input Source | Model / Backend Used | Actual Output Generated | Spoken Voice Response | Verification Category | Failure / Fallback Behavior |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Module 10: Object Location Memory** | *"Mera phone kahan hai?"* | `FIND_OBJECT` | `10_object_memory` | `ObjectMemoryAdapter.process_query()` -> `ObjectMemoryManager.get_latest_observation()` | Natural query + persistent SQLite DB | `ObjectMemoryDB` SQLite query on `data/memory/object_memory.db` | Historical observation record: `phone`, anchor: `dining table`, age: `3429 min` | *"Aapka phone abhi nahi dikh raha. Pichli baar lagbhag 3429 minute pehle dining table par dekha gaya tha."* | **REAL LOCAL EXECUTION (SQLite + Memory Manager)** | If object never recorded, distinguishes whether camera is active or inactive; never invents location. |
| **Module 11: Obstacle Awareness** | *"Kya rasta saaf hai?"* | `OBSTACLE_CHECK` | `11_obstacle_awareness` | `ObstacleService.process_query()` -> `ObstacleWarningGenerator.formulate_query_summary()` | Live evaluated perception hazards list | `ObstacleDetector` heuristic classifier + monocular depth geometry | Checked `_latest_hazards` and `_last_evaluated_time`; fresh vs stale state | *"Camera band ya inactive hai. Live perception ke bina rasta saaf hone ki pushti nahi ki ja sakti."* (or *"Saamne koi rukawat nahi dikh rahi..."* if camera active) | **REAL LOCAL EXECUTION (Service + Generator)** | If camera unavailable or frame timestamp > 5s stale, refuses to claim clear path; returns explicit perception warning. |
| **Module 12: Scene Understanding** | *"Mere aas paas kya hai?"* | `DESCRIBE_SCENE` | `12_scene_understanding` | `SceneService.process_query()` -> `SceneAnalyzer.analyze()` | Live detection buffer + spatial zoning | Multi-object spatial zone aggregator (`LEFT`, `CENTER`, `RIGHT`) | `SceneDescription` object with spatial relationships and counts | *"Aapke aas paas rasta saaf hai, koi vishesh cheez nahi dikh rahi."* (or multi-object spatial summary) | **REAL LOCAL EXECUTION (Scene Analyzer)** | If perception inactive, returns honest notice that scene understanding is inactive; never fabricates room items. |
| **Module 13: Alzheimer & Cognitive Assistance** | *"Main kahan hoon?"* | `HELP` (Orientation) | `13_alzheimer_assistance` | `AlzheimerService.process_query()` -> `OrientationGuide.get_orientation()` | System clock (`datetime.now()`) + `patient_profile.json` | Local temporal synthesizer + Hindi calendar converter | Day, date, time, and home address tuple | *"Aap apne ghar par hain. Aaj Mangalwaar (Tuesday), 24 November hai, aur abhi samay 03:19 AM hua hai."* | **REAL LOCAL EXECUTION (Orientation Engine)** | If profile missing, falls back to honest home prompt; never invents hospital, city, or event. |
| **Module 13: Familiar Person Identification** | *"Yeh kaun hai?"* | `IDENTIFY_PERSON` | `13_alzheimer_assistance` | `AlzheimerService.process_query()` -> `PatientProfileManager.get_person_by_name()` | Camera face crop or named target entity | `FaceDatabase` Fernet encrypted embeddings or configured profile directory | Sighting match or unknown person flag | *"Mujhe saamne koi jaana-pehchana vyakti nahi dikh raha. Kya main caregiver ko sampark karoon?"* | **REAL LOCAL EXECUTION (Profile Lookup)** | If person is unrecognized, explicitly states no familiar person in view; offers caregiver contact; never guesses. |
| **Module 14: Daily Assistance (List)** | *"Meri medicine kab leni hai?"* | `LIST_REMINDERS` | `14_daily_assistance` | `DailyService.process_query()` -> `ReminderDB.list_active()` | SQLite queries on `reminders.db` | SQLite3 database engine (WAL journal mode) | List of uncompleted `ReminderRecord` instances | *"Aapka abhi koi pending reminder nahi hai."* (or *"Aapke pass X reminder hain: [names]"*) | **REAL LOCAL EXECUTION (SQLite ReminderDB)** | If database corrupted, restores from `.corrupt.bak`, re-initializes tables, and speaks verbal recovery warning. |
| **Module 14: Daily Assistance (Set)** | *"Mujhe 10 minute baad paani peene ka reminder do"* | `SET_REMINDER` | `14_daily_assistance` | `DailyService.process_query()` -> `ReminderDB.add_reminder()` | Spoken offset parser + current epoch | SQLite3 database transaction with duplicate guard | Inserted reminder row ID, due epoch timestamp | *"Theek hai, maine Paani peene ka samay ke liye reminder set kar diya hai."* | **REAL LOCAL EXECUTION (SQLite ReminderDB)** | Deduplicates identical pending reminders within 60s; requires explicit user action to mark completed. |
| **Module 15: Safety & SOS (Trigger)** | *"Help me emergency hai"* | `TRIGGER_SOS` | `15_safety_sos` | `SafetyService.process_query()` -> `SOSManager.request_sos()` | User voice utterance | Threaded countdown timer (10s) + JSONL logger | State transition `IDLE -> PENDING_CONFIRMATION` | *"Kya aapko sach mein emergency madad chahiye? Haan bolein ya cancel karein."* | **REAL LOCAL EXECUTION (SOS State Machine)** | Preempts all lower-priority speech (< 100 ms); logs locally to `emergency_alerts.jsonl`. |
| **Module 15: Safety & SOS (Cancel)** | *"Nahi cancel kar do"* | `CANCEL_SOS` | `15_safety_sos` | `SafetyService.process_query()` -> `SOSManager.cancel_sos()` | User voice cancellation | State transition `PENDING_CONFIRMATION -> IDLE` | Timer cancelled; state reset | *"Emergency alert cancel kar diya gaya hai."* | **REAL LOCAL EXECUTION (SOS State Machine)** | Never claims user is safe without evidence ("Aap bilkul surakshit hain" removed). |
| **Module 17: Conversational Capabilities** | *"Aap kya madad kar sakti ho?"* | `HELP` | `17_assistant_orchestrator` | `AssistantOrchestrator._route_intent()` | Intent taxonomy and registered capabilities | Rule-based capability synthesizer | Formatted Hindi / English capability description | *"Main NAYRA hoon, aapki assistive companion. Aap mujhse cheezein dhundhne, saamne ka nazara dekhne, raste ki rukawat check karne, dawai reminder set karne, ya emergency alert ke liye keh sakte hain."* | **REAL LOCAL EXECUTION (Orchestrator)** | Never fabricates unverified caregiver presence or false reassurance. |

---

## 2. Distinction of Execution Modalities

To maintain total transparency and engineering integrity, each subsystem is classified into one of three execution categories:

### Category A: Real Local Model / Database Execution (Fully Verified)
- **Object Memory Manager (`10_object_memory`)**: Tested against active local SQLite database `data/memory/object_memory.db`.
- **Obstacle Service & Warning Generator (`11_obstacle_awareness`)**: Tested against real heuristic hazard geometry, urgency calculation, and zone classification.
- **Scene Understanding (`12_scene_understanding`)**: Tested against multi-zone spatial aggregation engine.
- **Cognitive Grounding & Orientation (`13_alzheimer_assistance`)**: Tested against local Python datetime conversion and `data/patient_profile.json`.
- **Reminder & Medication DB (`14_daily_assistance`)**: Tested against SQLite WAL transactions with corruption recovery and isolation.
- **Two-Stage Safety SOS (`15_safety_sos`)**: Tested against thread-safe state machine, priority speech preemption, and local JSONL logging.
- **YOLOv10n Inference (`03_object_detection`)**: Measured cold-start latency of 106.6 ms via CPUExecutionProvider ONNX runtime.
- **Kokoro-82M TTS (`08_tts`)**: Measured cold-start latency of 1,100.3 ms on local ONNX runtime.
- **Faster-Whisper Tiny STT (`09_voice`)**: Measured cold-start latency of 2,965.0 ms on CTranslate2 CPU engine.

### Category B: Software-Only Tested (Physical Hardware Emulated)
- **Microphone Stream (`09_voice`)**: WAV buffer feed tested via software mock; physical Android IP Webcam HTTP audio stream requires physical phone.
- **Webcam Stream (`01_camera_input`)**: Tested in `--no-camera` mode and synthetic frame buffers; live RTSP/HTTP Wi-Fi stream requires physical Android phone.

### Category C: Hardware Validation Pending (Explicitly Documented)
- **Wearable Smart Glasses Hardware**: Physical camera mount, battery thermals, and wearable form factor are pending future hardware development phases.
- **Outbound Cellular Telephony**: Automated GSM SMS/dialer hardware is pending; system truthfully operates via local JSONL event logging and voice prompts.
