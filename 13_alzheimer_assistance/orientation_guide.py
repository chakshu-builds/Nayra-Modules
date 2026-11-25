"""Orientation and reassurance guide for cognitive grounding."""

from __future__ import annotations

import datetime
from typing import Optional, Tuple


HINDI_DAYS = {
    0: "Somwaar (Monday)",
    1: "Mangalwaar (Tuesday)",
    2: "Budhwaar (Wednesday)",
    3: "Guruwaar (Thursday)",
    4: "Shukrawaar (Friday)",
    5: "Shaniwaar (Saturday)",
    6: "Raviwaar (Sunday)",
}

HINDI_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


class OrientationGuide:
    """Provides gentle, calming temporal and spatial orientation for Alzheimer's patients."""

    def __init__(self, home_location: str = "apne ghar par") -> None:
        self.home_location = home_location

    def get_orientation(self, now: datetime.datetime | None = None) -> Tuple[str, str]:
        """Generate reassuring time, date, and spatial orientation in Hindi and English."""
        if now is None:
            now = datetime.datetime.now()

        day_name = HINDI_DAYS.get(now.weekday(), "aaj")
        month_name = HINDI_MONTHS.get(now.month, "")
        time_str = now.strftime("%I:%M %p")

        spoken_hi = (
            f"Aap {self.home_location} hain. "
            f"Aaj {day_name}, {now.day} {month_name} hai, aur abhi samay {time_str} hua hai."
        )

        spoken_en = (
            f"You are at {self.home_location}. "
            f"Today is {now.strftime('%A, %B %d')}, and the current time is {time_str}."
        )

        return spoken_hi, spoken_en

    def get_reassurance(self, caregiver_name: Optional[str] = None) -> Tuple[str, str]:
        """Generate calming, truthful reassurance response without unsupported claims."""
        if caregiver_name and caregiver_name.strip():
            spoken_hi = (
                f"Main NAYRA hoon, main aapke saath hoon. Aapke registered caregiver {caregiver_name} hain. "
                "Agar aapko unki madad chahiye toh main emergency alert ya call lagane mein sahayata kar sakti hoon."
            )
            spoken_en = (
                f"I am NAYRA, I am with you. Your registered caregiver is {caregiver_name}. "
                "If you need assistance, I can help you contact them."
            )
        else:
            spoken_hi = "Main NAYRA hoon, main aapke saath hoon. Main yahan aapki sahayata ke liye maujood hoon."
            spoken_en = "I am NAYRA, I am with you. I am here to assist you."
        return spoken_hi, spoken_en
