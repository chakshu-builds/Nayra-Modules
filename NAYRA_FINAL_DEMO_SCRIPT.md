# NAYRA — Complete Final Demonstration Script

This document provides a scripted demonstration walkthrough showcasing all assistive capabilities of the unified NAYRA MVP.

---

## Demonstration Overview

| Step | Assistive Capability | Spoken User Query | Expected NAYRA Verbal Response (Hindi/English) | Internal Action | Mode / Testing |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Startup Greeting** | *(System start)* | *"NAYRA assistive companion taiyaar hai. Main aapki kya madad kar sakti hoon?"* | Topological initialization of Modules 01–18; priority speech queue active. | REAL MODELS / TEXT DEBUG |
| **2** | **Orientation (Alzheimer's)** | *"Main kahan hoon?"* | *"Aap apne ghar par hain. Aaj Mangalwaar (Tuesday), 24 November hai, aur abhi samay 02:55 AM hua hai."* | Module 13 queries local patient profile and temporal clock; delivers calming reassurance without false safety assertions. | REAL MODULE / VERIFIED |
| **3** | **Person Identification** | *"Yeh kaun hai?"* | *"Mujhe saamne koi jaana-pehchana vyakti nahi dikh raha. Kya main caregiver ko sampark karoon?"* *(or identifies Rahul if face enrolled)* | Module 13 & Module 06 resolve face embedding against database; never invents identities. | REAL MODULE / VERIFIED |
| **4** | **Obstacle Path Check** | *"Kya saamne rasta saaf hai?"* | *"Aapke saamne koi rukawat nahi dikh rahi, rasta saaf lag raha hai."* *(or "Camera band ya inactive hai. Live perception ke bina rasta saaf hone ki pushti nahi ki ja sakti" if camera off)* | Module 11 checks center zone proximity, clearance, and frame freshness. | REAL MODULE / CAMERA FEED |
| **5** | **Scene Understanding** | *"Mere saamne kya hai?"* | *"Saamne table par laptop rakha hai, aur right side mein ek kursi hai."* *(or "Aapke saamne koi vishesh cheez nahi dikh rahi hai")* | Module 12 analyzes visual detections and spatial anchors across zones. | REAL MODULE / CAMERA FEED |
| **6** | **Object Location Memory** | *"Mera phone kahan hai?"* | *"Aapka phone saamne dining table par hai."* *(or "Pichli baar lagbhag X minute pehle dining table par dekha gaya tha")* | Module 10 queries persistent episodic SQLite database with furniture anchors; 4 truthful visibility states. | REAL MODULE / VERIFIED |
| **7** | **Read Aloud (OCR)** | *"Board par kya likha hai?"* | *"Saamne likha hai: EMERGENCY EXIT."* *(or "Saamne koi text ya likhawat nahi dikh rahi hai")* | Module 07 extracts and speaks text from active camera frame. | REAL MODULE / CAMERA FEED |
| **8** | **Set Medication Reminder**| *"Mujhe 10 minute baad dawai lene ki yaad dilana"*| *"Theek hai, maine Dawai lene ka samay ke liye reminder set kar diya hai."* | Module 14 writes scheduled event to `data/reminders.db` and starts scheduler countdown. | REAL MODULE / PERSISTENT SQLITE |
| **9** | **List Reminders** | *"Meri medicine kab leni hai?"* | *"Aapke pass 1 reminder hai: Dawai lene ka samay."* *(or "Aapka abhi koi pending reminder nahi hai")* | Module 14 queries pending active records from SQLite database. | REAL MODULE / PERSISTENT SQLITE |
| **10**| **Emergency SOS Trigger** | *"Help me emergency hai!"* | *"Kya aapko sach mein emergency madad chahiye? Haan bolein ya cancel karein."* | Module 15 preempts current speech with `EMERGENCY_INTERACTION` and starts 10s confirmation timer. | REAL MODULE / PREEMPTION VERIFIED |
| **11**| **Emergency SOS Confirm** | *"Haan madad chahiye"* | *"Emergency alert activate ho gaya hai aur local log mein darj ho gaya hai. Kripya caregiver Rahul ko call karne ke liye phone ready rakhein."* | Module 15 writes payload to `data/emergency_alerts.jsonl`; truthful local announcement without false cellular dispatch claims. | REAL MODULE / AUDIT LOG |
| **12**| **Emergency SOS Cancel (Alt)**| *"Cancel kar do galti se bol diya"*| *"Emergency alert cancel kar diya gaya hai."* | Module 15 cancels active timer and resets state to IDLE without unfounded safety assertions. | REAL MODULE / VERIFIED |
| **13**| **Capabilities & Help** | *"Aap kya madad kar sakti ho?"* | *"Main NAYRA hoon, aapki assistive companion. Aap mujhse cheezein dhundhne, saamne ka nazara dekhne, raste ki rukawat check karne, dawai reminder set karne, ya emergency alert ke liye keh sakte hain."* | Module 17 orchestrator provides truthful capability summary without fabricated names. | REAL MODULE / VERIFIED |

---

## Live Running Instructions

To execute this exact demonstration interactively in terminal text-voice mode:
```bash
python main.py --no-camera --text-debug --lang hi
```
To run the automated scripted verification test:
```bash
python main.py --smoke-test --no-camera
```
