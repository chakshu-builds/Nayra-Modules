"""Patient profile and familiar faces management for Alzheimer's care."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NAYRA-PatientProfile")


@dataclass
class FamiliarPerson:
    """Record of a familiar family member, caregiver, or doctor."""
    person_id: str
    name: str
    relationship: str
    phone: str = ""
    notes: str = ""
    face_embedding: Optional[List[float]] = None


class PatientProfileManager:
    """Manages patient profile, home environment details, and familiar person directory."""

    def __init__(self, data_file: Optional[Path] = None) -> None:
        if data_file is None:
            data_file = Path(__file__).resolve().parents[1] / "data" / "patient_profile.json"
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        self.patient_name: str = "User"
        self.home_address: str = "apne ghar par"
        self.primary_caregiver: str = "Rahul"
        self.caregiver_phone: str = "9876543210"
        self.familiar_persons: Dict[str, FamiliarPerson] = {}
        self.load()

    def load(self) -> None:
        """Load profile from disk or initialize defaults."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.patient_name = data.get("patient_name", "User")
                self.home_address = data.get("home_address", "Apne ghar par")
                self.primary_caregiver = data.get("primary_caregiver", "Rahul")
                self.caregiver_phone = data.get("caregiver_phone", "9876543210")
                self.familiar_persons = {}
                for pid, pdata in data.get("familiar_persons", {}).items():
                    self.familiar_persons[pid] = FamiliarPerson(**pdata)
                logger.info(f"Loaded patient profile for {self.patient_name} with {len(self.familiar_persons)} familiar persons.")
                return
            except Exception as e:
                logger.warning(f"Failed to load patient profile ({e}), using default template.")

        # Default fallback data
        self.familiar_persons = {
            "p1": FamiliarPerson(
                person_id="p1",
                name="Rahul",
                relationship="beta",
                phone="9876543210",
                notes="Roz shaam ko aate hain.",
            ),
            "p2": FamiliarPerson(
                person_id="p2",
                name="Priya",
                relationship="caregiver",
                phone="9812345678",
                notes="Dawai aur khane ka dhyan rakhti hain.",
            ),
        }
        self.save()

    def save(self) -> None:
        """Save profile to disk."""
        try:
            payload = {
                "patient_name": self.patient_name,
                "home_address": self.home_address,
                "primary_caregiver": self.primary_caregiver,
                "caregiver_phone": self.caregiver_phone,
                "familiar_persons": {pid: asdict(p) for pid, p in self.familiar_persons.items()},
            }
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving patient profile: {e}")

    def get_person_by_name(self, name: str) -> Optional[FamiliarPerson]:
        name_lower = name.strip().lower()
        for p in self.familiar_persons.values():
            if p.name.lower() == name_lower:
                return p
        return None

    def add_or_update_person(self, person: FamiliarPerson) -> None:
        self.familiar_persons[person.person_id] = person
        self.save()
