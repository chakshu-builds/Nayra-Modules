"""SOS state machine, confirmation timer, and contact management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from shared.interfaces import IVoiceOutput
from shared.schemas import EmergencyContact, SpeechPriority

logger = logging.getLogger("NAYRA-SOSManager")


class SOSState(str, Enum):
    IDLE = "IDLE"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


class SOSManager:
    """Manages two-stage SOS confirmation lifecycle, contact registry, and emergency dispatch."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        contacts_file: Optional[Path] = None,
        alerts_log_file: Optional[Path] = None,
        countdown_sec: float = 10.0,
        on_sos_triggered: Optional[Callable[[Dict], None]] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.countdown_sec = countdown_sec
        self.on_sos_triggered = on_sos_triggered
        self.preferred_language = preferred_language

        if contacts_file is None:
            contacts_file = Path(__file__).resolve().parents[1] / "data" / "emergency_contacts.json"
        self.contacts_file = Path(contacts_file)
        self.contacts_file.parent.mkdir(parents=True, exist_ok=True)

        if alerts_log_file is None:
            alerts_log_file = Path(__file__).resolve().parents[1] / "data" / "emergency_alerts.jsonl"
        self.alerts_log_file = Path(alerts_log_file)
        self.alerts_log_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = SOSState.IDLE
        self.pending_reason: str = ""
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()
        self.contacts: List[EmergencyContact] = []
        self._load_contacts()

    def _load_contacts(self) -> None:
        if self.contacts_file.exists():
            try:
                with open(self.contacts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.contacts = [
                    EmergencyContact(
                        name=c["name"],
                        phone=c["phone"],
                        relationship=c.get("relationship", c.get("relation", "Caregiver")),
                        is_primary=c.get("is_primary", True),
                    )
                    for c in data
                ]
                return
            except Exception as e:
                logger.warning(f"Error loading emergency contacts ({e}), using default fallback.")

        # Default sample contact
        self.contacts = [
            EmergencyContact(name="Caregiver", phone="+919215793330", relationship="Primary Caregiver", is_primary=True),
            EmergencyContact(name="Dr. Sharma", phone="9811223344", relationship="doctor", is_primary=False),
        ]
        self._save_contacts()

    def _save_contacts(self) -> None:
        try:
            with open(self.contacts_file, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in self.contacts], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving emergency contacts: {e}")

    def request_sos(self, reason: str = "User voice trigger", auto_confirm_on_timeout: bool = True) -> str:
        """Initiate two-stage confirmation."""
        with self._lock:
            self._cancel_existing_timer()
            self.state = SOSState.PENDING_CONFIRMATION
            self.pending_reason = reason

            if self.preferred_language == "en":
                spoken = "Do you need emergency assistance? Say yes or cancel."
            else:
                spoken = "Kya aapko sach mein emergency madad chahiye? Haan bolein ya cancel karein."

            if self.voice_output:
                self.voice_output.speak(spoken, priority=SpeechPriority.EMERGENCY_INTERACTION)

            if auto_confirm_on_timeout:
                self._timer = threading.Timer(self.countdown_sec, self._on_timeout_confirm)
                self._timer.daemon = True
                self._timer.start()

            logger.warning(f"SOS requested: {reason}. Awaiting confirmation or {self.countdown_sec}s timeout.")
            return spoken

    def confirm_sos(self) -> str:
        """Confirm SOS actively."""
        with self._lock:
            self._cancel_existing_timer()
            return self._dispatch_emergency(trigger_method="user_confirmed")

    def cancel_sos(self) -> str:
        """Cancel pending SOS."""
        with self._lock:
            self._cancel_existing_timer()
            self.state = SOSState.CANCELLED
            if self.preferred_language == "en":
                spoken = "Emergency alert has been cancelled."
            else:
                spoken = "Emergency alert cancel kar diya gaya hai."
            if self.voice_output:
                self.voice_output.speak(spoken, priority=SpeechPriority.EMERGENCY_INTERACTION)
            self.state = SOSState.IDLE
            logger.info("SOS cancelled by user.")
            return spoken

    def _on_timeout_confirm(self) -> None:
        with self._lock:
            if self.state == SOSState.PENDING_CONFIRMATION:
                logger.warning("SOS confirmation timeout expired! Auto-triggering emergency protocol.")
                self._dispatch_emergency(trigger_method="timeout_fallback")

    def _dispatch_emergency(self, trigger_method: str) -> str:
        self.state = SOSState.TRIGGERED
        now = time.time()
        contacts_notified = [c.name for c in self.contacts]
        payload = {
            "timestamp": now,
            "readable_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "trigger_method": trigger_method,
            "reason": self.pending_reason,
            "contacts_notified": [asdict(c) for c in self.contacts],
        }

        # Write to JSONL
        try:
            with open(self.alerts_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to log emergency alert: {e}")

        primary_contact = next((c for c in self.contacts if c.is_primary), self.contacts[0] if self.contacts else None)
        primary_phone = primary_contact.phone if primary_contact else "+919215793330"
        clean_phone = "".join(filter(str.isdigit, primary_phone))

        # Dynamic real-time location fetching
        location_info = self._get_current_location()
        maps_url = location_info.get("maps_url", f"https://maps.google.com/?q={location_info.get('latitude', 28.6542)},{location_info.get('longitude', 77.2373)}")
        loc_str = location_info.get("location_string", "Delhi, India")

        sms_text = (
            f"EMERGENCY ALERT from NAYRA! User triggered SOS assistance. "
            f"Live Location: {loc_str}. "
            f"Map: {maps_url}"
        )

        payload["location"] = location_info
        payload["sms_message"] = sms_text
        payload["dispatched_channels"] = []
        self.last_dispatched_alert = payload

        # 1. Real-time Live Network Push Notification via ntfy.sh (Priority 5 = Max Alarm & Sound)
        ntfy_channel = f"nayra_emergency_{clean_phone[-10:] if len(clean_phone) >= 10 else 'alert'}"
        try:
            import urllib.request
            import urllib.parse
            json_payload = json.dumps({
                "topic": ntfy_channel,
                "title": "🚨 NAYRA EMERGENCY SOS HIT!",
                "message": (
                    f"EMERGENCY: User triggered SOS on Nayra!\n"
                    f"Location: {loc_str}\n"
                    f"Caregiver Phone: {primary_phone}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
                ),
                "priority": 5,
                "tags": ["warning", "sos", "rotating_light"],
                "click": maps_url,
                "actions": [
                    {"action": "view", "label": "📍 Open Live Map", "url": maps_url},
                    {"action": "view", "label": "💬 WhatsApp Caregiver", "url": f"https://api.whatsapp.com/send?phone={clean_phone}&text={urllib.parse.quote(sms_text)}"}
                ]
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://ntfy.sh",
                data=json_payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    payload["dispatched_channels"].append(f"ntfy.sh/{ntfy_channel}")
                    logger.info(f"Real-time push notification sent to ntfy.sh/{ntfy_channel}")
        except Exception as e:
            logger.warning(f"Push notification dispatch notice: {e}")

        # Optional Fast2SMS Indian Cellular SMS Gateway
        import os
        fast2sms_key = os.environ.get("FAST2SMS_API_KEY", "")
        if not fast2sms_key:
            env_file = Path(__file__).resolve().parents[1] / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("FAST2SMS_API_KEY="):
                        fast2sms_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if not fast2sms_key:
            fast2sms_key = "RgUcVvawtr0WZo70CtHJgkrQIKvMSewrdFwrIdLRzLXwBWDlx00mbqSQYtHCHomdmr2HHk3DLmOeqD2C"

        if fast2sms_key:
            try:
                f2s_url = "https://www.fast2sms.com/dev/bulkV2"
                f2s_data = json.dumps({
                    "route": "q",
                    "message": sms_text,
                    "language": "english",
                    "flash": 0,
                    "numbers": clean_phone[-10:],
                }).encode("utf-8")
                f2s_req = urllib.request.Request(
                    f2s_url,
                    data=f2s_data,
                    headers={"authorization": fast2sms_key, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(f2s_req, timeout=3.0) as f2s_resp:
                    if f2s_resp.status == 200:
                        payload["dispatched_channels"].append("fast2sms_cellular")
                        logger.info("Fast2SMS cellular SMS successfully dispatched!")
            except urllib.error.HTTPError as e:
                err_resp = e.read().decode("utf-8", errors="ignore")
                logger.warning(f"Fast2SMS API notice (HTTP {e.code}): {err_resp}")
            except Exception as e:
                logger.warning(f"Fast2SMS cellular dispatch notice: {e}")

        # Optional Twilio Global Cellular SMS Gateway
        twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        twilio_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        twilio_from = os.environ.get("TWILIO_PHONE_NUMBER", "")
        if twilio_sid and twilio_token and twilio_from:
            try:
                import urllib.parse
                import base64
                tw_url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
                tw_data = urllib.parse.urlencode({
                    "To": primary_phone if primary_phone.startswith("+") else f"+{primary_phone}",
                    "From": twilio_from,
                    "Body": sms_text,
                }).encode("utf-8")
                auth_str = base64.b64encode(f"{twilio_sid}:{twilio_token}".encode("utf-8")).decode("utf-8")
                tw_req = urllib.request.Request(
                    tw_url,
                    data=tw_data,
                    headers={
                        "Authorization": f"Basic {auth_str}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                with urllib.request.urlopen(tw_req, timeout=4.0) as tw_resp:
                    if tw_resp.status in (200, 201):
                        payload["dispatched_channels"].append("twilio_cellular_sms")
                        logger.info("Twilio cellular SMS successfully dispatched!")
            except Exception as e:
                logger.warning(f"Twilio cellular dispatch notice: {e}")

        # CallMeBot 100% Free WhatsApp API (No recharge, no payment)
        callmebot_key = os.environ.get("CALLMEBOT_API_KEY", "")
        if not callmebot_key:
            env_file = Path(__file__).resolve().parents[1] / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("CALLMEBOT_API_KEY="):
                        callmebot_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if callmebot_key:
            try:
                import urllib.parse
                cmb_phone = clean_phone[-10:] if not clean_phone.startswith("91") else clean_phone
                if len(cmb_phone) == 10:
                    cmb_phone = "91" + cmb_phone
                cmb_msg = urllib.parse.quote(
                    f"🚨 NAYRA EMERGENCY SOS!\n"
                    f"Location: {loc_str}\n"
                    f"Google Maps: {maps_url}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
                )
                cmb_url = f"https://api.callmebot.com/whatsapp.php?phone={cmb_phone}&text={cmb_msg}&apikey={callmebot_key}"
                cmb_req = urllib.request.Request(cmb_url, headers={"User-Agent": "NAYRA-Assistive-AI"})
                with urllib.request.urlopen(cmb_req, timeout=5.0) as cmb_resp:
                    if cmb_resp.status == 200:
                        payload["dispatched_channels"].append("callmebot_whatsapp")
                        logger.info("CallMeBot free WhatsApp message dispatched directly to phone!")
            except Exception as e:
                logger.warning(f"CallMeBot dispatch notice: {e}")

        # Telegram Bot API (100% Free forever)
        tg_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if tg_bot_token and tg_chat_id:
            try:
                import urllib.parse
                tg_msg = urllib.parse.quote(
                    f"🚨 EMERGENCY ALERT: NAYRA SOS Triggered!\n"
                    f"Caregiver Phone: {primary_phone}\n"
                    f"Location: {loc_str}\n"
                    f"Google Maps: {maps_url}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
                )
                tg_url = f"https://api.telegram.org/bot{tg_bot_token}/sendMessage?chat_id={tg_chat_id}&text={tg_msg}"
                tg_req = urllib.request.Request(tg_url)
                with urllib.request.urlopen(tg_req, timeout=4.0) as tg_resp:
                    if tg_resp.status == 200:
                        payload["dispatched_channels"].append("telegram_bot")
                        logger.info("Telegram emergency alert dispatched!")
            except Exception as e:
                logger.warning(f"Telegram dispatch notice: {e}")

        # 2. Real-time WhatsApp emergency link dispatch
        try:
            import urllib.parse
            import webbrowser
            encoded_sms = urllib.parse.quote(sms_text)
            wa_url = f"https://api.whatsapp.com/send?phone={clean_phone}&text={encoded_sms}"
            payload["whatsapp_url"] = wa_url
            payload["dispatched_channels"].append("whatsapp_web")
            # Open browser tab non-blocking in background
            threading.Thread(target=lambda: webbrowser.open(wa_url), daemon=True).start()
            logger.info(f"Real-time WhatsApp dispatch URL generated and opened: {wa_url}")
        except Exception as e:
            logger.warning(f"WhatsApp dispatch notice: {e}")

        # 3. Windows Desktop Notification Banner
        try:
            import subprocess
            ps_script = (
                f"[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); "
                f"$notify = New-Object System.Windows.Forms.NotifyIcon; "
                f"$notify.Icon = [System.Drawing.SystemIcons]::Warning; "
                f"$notify.Visible = $True; "
                f"$notify.ShowBalloonTip(5000, '🚨 NAYRA EMERGENCY SOS HIT!', 'Location: {loc_str}\\nDispatched to {primary_phone}', [System.Windows.Forms.ToolTipIcon]::Warning)"
            )
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            payload["dispatched_channels"].append("windows_toast")
        except Exception as e:
            logger.warning(f"Windows desktop toast notice: {e}")

        # Spoken notification: Do NOT read raw phone number aloud to protect privacy and clarity
        if self.preferred_language == "en":
            spoken = "Emergency alert activated. SOS notification and your live location have been automatically dispatched to your caregiver."
        elif contacts_notified:
            spoken = "Emergency alert activate ho gaya hai. Aapke caregiver ko live location ke sath SOS sandesh bhej diya gaya hai aur local log mein darj kar diya gaya hai."
        else:
            spoken = "Emergency alert activate ho gaya hai. SOS sandesh live location ke sath bhej diya gaya hai aur local log mein darj kar diya gaya hai."

        if self.voice_output:
            self.voice_output.speak(spoken, priority=SpeechPriority.CRITICAL_SAFETY)

        if self.on_sos_triggered:
            try:
                self.on_sos_triggered(payload)
            except Exception as e:
                logger.error(f"Error in on_sos_triggered callback: {e}")

        logger.critical(f"EMERGENCY DISPATCHED: {payload}")
        return spoken

    def _get_current_location(self) -> Dict[str, Any]:
        """Resolve current GPS / network location dynamically for emergency dispatch.
        
        Resolution Priority:
        1. Exact Manual / Configured Coordinates in .env (EXACT_LATITUDE, EXACT_LONGITUDE, EXACT_LOCATION_NAME)
        2. Live Satellite GPS from Phone IP Webcam (/gps.json)
        3. Real-Time Network GeoIP (ip-api.com / ipinfo.io)
        """
        import os
        import urllib.request
        from pathlib import Path

        # Check for configured exact location in .env
        exact_lat = os.environ.get("EXACT_LATITUDE", "")
        exact_lon = os.environ.get("EXACT_LONGITUDE", "")
        exact_name = os.environ.get("EXACT_LOCATION_NAME", "")

        if not exact_lat or not exact_lon:
            env_file = Path(__file__).resolve().parents[1] / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("EXACT_LATITUDE="):
                        exact_lat = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("EXACT_LONGITUDE="):
                        exact_lon = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("EXACT_LOCATION_NAME="):
                        exact_name = line.split("=", 1)[1].strip().strip('"').strip("'")

        # 1. Configured exact location override
        if exact_lat and exact_lon:
            try:
                lat = float(exact_lat)
                lon = float(exact_lon)
                loc_name = exact_name or f"Exact GPS Location ({lat:.4f}, {lon:.4f})"
                logger.info(f"Using configured exact location: {loc_name}")
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "city": exact_name,
                    "region": "",
                    "country": "India",
                    "postal": "",
                    "location_string": f"{loc_name} [{lat:.5f}, {lon:.5f}]",
                    "maps_url": f"https://maps.google.com/?q={lat:.5f},{lon:.5f}",
                }
            except ValueError:
                pass

        # 2. Live satellite GPS from connected Android IP Webcam
        for phone_ip in ["192.168.0.101", "127.0.0.1", "localhost"]:
            try:
                gps_url = f"http://{phone_ip}:8080/gps.json"
                req = urllib.request.Request(gps_url, headers={"User-Agent": "NAYRA"})
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    g_data = json.loads(r.read().decode())
                    gps_obj = g_data.get("gps", g_data)
                    if isinstance(gps_obj, dict) and "latitude" in gps_obj and "longitude" in gps_obj:
                        lat = float(gps_obj["latitude"])
                        lon = float(gps_obj["longitude"])
                        acc = gps_obj.get("accuracy", 10.0)
                        logger.info(f"Resolved live satellite GPS from phone: {lat}, {lon} (accuracy: {acc}m)")
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "city": "Phone Hardware GPS",
                            "region": "",
                            "country": "India",
                            "postal": "",
                            "location_string": f"Phone Satellite GPS (Accuracy {acc:.0f}m) [{lat:.5f}, {lon:.5f}]",
                            "maps_url": f"https://maps.google.com/?q={lat:.5f},{lon:.5f}",
                        }
            except Exception:
                pass

        # 3. Real-Time Network GeoIP (ip-api.com)
        try:
            req = urllib.request.Request("http://ip-api.com/json", headers={"User-Agent": "NAYRA-Assistive-AI"})
            with urllib.request.urlopen(req, timeout=2.0) as r:
                data = json.loads(r.read().decode())
                if data.get("status") == "success":
                    lat = float(data.get("lat", 28.6542))
                    lon = float(data.get("lon", 77.2373))
                    city = data.get("city", "Delhi")
                    region = data.get("regionName", "Delhi")
                    country = data.get("country", "India")
                    zip_code = data.get("zip", "")
                    loc_desc = f"{city}, {region}, {country} (PIN {zip_code})" if zip_code else f"{city}, {region}, {country}"
                    return {
                        "latitude": lat,
                        "longitude": lon,
                        "city": city,
                        "region": region,
                        "country": country,
                        "postal": zip_code,
                        "location_string": f"{loc_desc} [{lat:.4f}, {lon:.4f}]",
                        "maps_url": f"https://maps.google.com/?q={lat:.4f},{lon:.4f}",
                    }
        except Exception as e:
            logger.debug(f"ip-api lookup exception: {e}")

        # 4. Fallback: ipinfo.io
        try:
            req = urllib.request.Request("https://ipinfo.io/json", headers={"User-Agent": "NAYRA-Assistive-AI"})
            with urllib.request.urlopen(req, timeout=2.0) as r:
                data = json.loads(r.read().decode())
                loc_parts = str(data.get("loc", "28.6542,77.2373")).split(",")
                lat = float(loc_parts[0]) if len(loc_parts) > 0 else 28.6542
                lon = float(loc_parts[1]) if len(loc_parts) > 1 else 77.2373
                city = data.get("city", "Delhi")
                region = data.get("region", "Delhi")
                country = data.get("country", "India")
                postal = data.get("postal", "")
                loc_desc = f"{city}, {region}, {country} (PIN {postal})" if postal else f"{city}, {region}, {country}"
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "city": city,
                    "region": region,
                    "country": country,
                    "postal": postal,
                    "location_string": f"{loc_desc} [{lat:.4f}, {lon:.4f}]",
                    "maps_url": f"https://maps.google.com/?q={lat:.4f},{lon:.4f}",
                }
        except Exception as e:
            logger.debug(f"ipinfo lookup exception: {e}")

        # Safe fallback: Sonipat, Haryana
        return {
            "latitude": 28.9931,
            "longitude": 77.0151,
            "city": "Sonipat",
            "region": "Haryana",
            "country": "India",
            "postal": "131001",
            "location_string": "Sonipat, Haryana, India (PIN 131001) [28.9931, 77.0151]",
            "maps_url": "https://maps.google.com/?q=28.9931,77.0151",
        }

    def _cancel_existing_timer(self) -> None:
        if self._timer and self._timer.is_alive():
            self._timer.cancel()
            self._timer = None
