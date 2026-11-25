"""Consent-based local face database with authenticated symmetric encryption."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings
from recognition_types import FaceProfile

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Encrypted local storage for consented familiar person profiles and embeddings."""

    def __init__(self, settings: PersonRecognitionSettings | None = None) -> None:
        self.settings = settings or get_default_person_recognition_settings()
        self.db_path = Path(self.settings.db_path)
        self.key_path = Path(self.settings.key_path)

        self._cipher = self._init_cipher()
        self._profiles: dict[str, FaceProfile] = {}  # profile_id -> FaceProfile
        self._name_index: dict[str, str] = {}        # lowercase name -> profile_id

        self.load()

    def _init_cipher(self) -> Fernet:
        """Resolve encryption key from environment variable or key file."""
        env_key = os.getenv("NAYRA_ENCRYPTION_KEY")
        if env_key:
            try:
                return Fernet(env_key.encode("utf-8"))
            except Exception as e:
                logger.warning("Invalid NAYRA_ENCRYPTION_KEY environment variable (%s). Falling back to key file.", e)

        if self.key_path.exists():
            try:
                key_bytes = self.key_path.read_bytes().strip()
                return Fernet(key_bytes)
            except Exception as e:
                logger.error("Failed to read encryption key file %s: %s", self.key_path, e)

        # Generate a new secure key and write to disk
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        new_key = Fernet.generate_key()
        try:
            self.key_path.write_bytes(new_key)
            try:
                os.chmod(self.key_path, 0o600)
            except (AttributeError, OSError):
                pass
            logger.info("Generated new local face database encryption key at %s", self.key_path)
        except Exception as e:
            logger.error("Failed to write encryption key: %s", e)

        return Fernet(new_key)

    def load(self) -> None:
        """Load and decrypt profiles from persistent disk storage."""
        self._profiles.clear()
        self._name_index.clear()

        if not self.db_path.exists():
            return

        try:
            encrypted_data = self.db_path.read_bytes()
            if not encrypted_data:
                return

            decrypted_bytes = self._cipher.decrypt(encrypted_data)
            data_dict = json.loads(decrypted_bytes.decode("utf-8"))

            profiles_list = data_dict.get("profiles", [])
            for p_dict in profiles_list:
                # Validate model architecture and embedding dimension
                if p_dict.get("embedding_dim", 128) != 128:
                    logger.warning("Skipping profile '%s' with incompatible embedding dim.", p_dict.get("display_name"))
                    continue

                profile = FaceProfile(
                    profile_id=p_dict["profile_id"],
                    display_name=p_dict["display_name"],
                    embeddings=p_dict["embeddings"],
                    enrolled_at=p_dict.get("enrolled_at", 0.0),
                    sample_count=len(p_dict["embeddings"]),
                    model_name=p_dict.get("model_name", "sface_2021dec"),
                    embedding_dim=p_dict.get("embedding_dim", 128),
                )
                self._profiles[profile.profile_id] = profile
                self._name_index[profile.display_name.lower().strip()] = profile.profile_id

            logger.info("Loaded %d enrolled person profile(s) from local database.", len(self._profiles))
        except InvalidToken:
            logger.error("Failed to decrypt face database at %s. Key mismatch or data corrupted.", self.db_path)
        except Exception as err:
            logger.error("Error reading face database: %s", err)

    def save(self) -> None:
        """Serialize and encrypt profiles to disk atomically."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "model_name": "sface_2021dec",
            "embedding_dim": 128,
            "profiles": [
                {
                    "profile_id": p.profile_id,
                    "display_name": p.display_name,
                    "embeddings": p.embeddings,
                    "enrolled_at": p.enrolled_at,
                    "sample_count": len(p.embeddings),
                    "model_name": p.model_name,
                    "embedding_dim": p.embedding_dim,
                }
                for p in self._profiles.values()
            ],
        }

        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        encrypted_data = self._cipher.encrypt(json_bytes)

        # Atomic write via temporary file
        temp_path = self.db_path.with_suffix(".tmp")
        temp_path.write_bytes(encrypted_data)
        temp_path.replace(self.db_path)

    def enroll_person(
        self,
        display_name: str,
        embeddings: Sequence[np.ndarray | list[float]],
    ) -> FaceProfile:
        """Enroll a person with validated reference embeddings under explicit user consent."""
        clean_name = display_name.strip()
        if not clean_name:
            raise ValueError("Display name cannot be empty.")

        if len(embeddings) < self.settings.min_samples_per_enrollment:
            raise ValueError(
                f"Requires at least {self.settings.min_samples_per_enrollment} face samples, got {len(embeddings)}."
            )

        norm_embeddings: list[list[float]] = []
        for emb in embeddings[: self.settings.max_samples_per_enrollment]:
            arr = np.asarray(emb, dtype=np.float32).flatten()
            if arr.shape[0] != 128:
                raise ValueError(f"Expected 128-dimensional embedding, got {arr.shape[0]}.")
            norm = np.linalg.norm(arr)
            if norm > 1e-6:
                arr = arr / norm
            norm_embeddings.append(arr.tolist())

        key = clean_name.lower()
        if key in self._name_index:
            # Update existing profile
            profile_id = self._name_index[key]
            profile = self._profiles[profile_id]
            profile.embeddings = norm_embeddings
            profile.sample_count = len(norm_embeddings)
            logger.info("Updated existing profile '%s' (ID: %s).", clean_name, profile_id)
        else:
            profile_id = str(uuid.uuid4())
            profile = FaceProfile(
                profile_id=profile_id,
                display_name=clean_name,
                embeddings=norm_embeddings,
                sample_count=len(norm_embeddings),
            )
            self._profiles[profile_id] = profile
            self._name_index[key] = profile_id
            logger.info("Enrolled new familiar person '%s' (ID: %s).", clean_name, profile_id)

        self.save()
        return profile

    def get_profile(self, profile_id: str) -> FaceProfile | None:
        return self._profiles.get(profile_id)

    def get_profile_by_name(self, display_name: str) -> FaceProfile | None:
        pid = self._name_index.get(display_name.lower().strip())
        return self._profiles.get(pid) if pid else None

    def list_profiles(self) -> list[FaceProfile]:
        return list(self._profiles.values())

    def delete_profile(self, identifier: str) -> bool:
        """Permanently delete an enrolled person's complete profile and biometric embeddings."""
        clean_id = identifier.strip()
        target_id: str | None = None

        if clean_id in self._profiles:
            target_id = clean_id
        else:
            target_id = self._name_index.get(clean_id.lower())

        if target_id and target_id in self._profiles:
            p = self._profiles.pop(target_id)
            self._name_index.pop(p.display_name.lower().strip(), None)
            self.save()
            logger.info("Deleted profile '%s' (ID: %s) and erased all biometric embeddings.", p.display_name, target_id)
            return True

        return False

    def clear_all(self) -> None:
        """Erase all enrolled profiles."""
        self._profiles.clear()
        self._name_index.clear()
        self.save()
