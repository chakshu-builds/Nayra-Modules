"""Consent-based familiar person enrollment workflow and management CLI."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings
from face_aligner import FaceAligner
from face_database import FaceDatabase
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from recognition_types import FaceProfile

logger = logging.getLogger(__name__)


def enroll_from_images(
    display_name: str,
    image_paths: list[str | Path],
    settings: PersonRecognitionSettings | None = None,
    database: FaceDatabase | None = None,
) -> FaceProfile:
    """Enroll a familiar person from consented reference image files."""
    settings = settings or get_default_person_recognition_settings()
    database = database or FaceDatabase(settings)

    detector = FaceDetector(settings)
    aligner = FaceAligner()
    recognizer = FaceRecognizer(settings)

    valid_embeddings: list[np.ndarray] = []

    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Image path does not exist: %s", path)
            continue

        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Could not read image: %s", path)
            continue

        detections = detector.detect(img)
        if len(detections) != 1:
            logger.warning("Expected exactly 1 face in reference image %s, found %d", path.name, len(detections))
            continue

        aligned = aligner.align(img, detections[0])
        if aligned is None:
            continue

        valid, msg = aligner.check_quality(aligned)
        if not valid:
            logger.warning("Quality check failed for %s: %s", path.name, msg)
            continue

        emb = recognizer.extract_embedding(aligned)
        if emb is not None:
            valid_embeddings.append(emb)

    if len(valid_embeddings) < settings.min_samples_per_enrollment:
        raise ValueError(
            f"Insufficient valid face samples ({len(valid_embeddings)}). "
            f"Required: {settings.min_samples_per_enrollment}."
        )

    profile = database.enroll_person(display_name, valid_embeddings)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="NAYRA - Consent-Based Familiar Person Enrollment")
    parser.add_argument("--list", action="store_true", help="List all locally enrolled familiar persons.")
    parser.add_argument("--enroll", type=str, default=None, help="Name of person to enroll.")
    parser.add_argument("--images", nargs="+", help="Image files for enrollment.")
    parser.add_argument("--relationship", type=str, default="", help="Relationship (e.g. Sister, Friend, Caregiver).")
    parser.add_argument("--delete", type=str, default=None, help="Name or ID of person profile to delete.")
    parser.add_argument("--clear-all", action="store_true", help="Delete all enrolled profiles.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    settings = get_default_person_recognition_settings()
    db = FaceDatabase(settings)

    from context_database import PersonContextDatabase
    context_db = PersonContextDatabase(settings)

    if args.list:
        profiles = db.list_profiles()
        print(f"\n--- Enrolled Familiar Persons ({len(profiles)}) ---")
        if not profiles:
            print("No profiles enrolled. Use --enroll to add trusted individuals.")
        for p in profiles:
            enrolled_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.enrolled_at))
            ctx = context_db.get_context(p.profile_id)
            rel_str = f" [{ctx.relationship}]" if ctx and ctx.relationship else ""
            print(f" • [{p.profile_id[:8]}] {p.display_name}{rel_str} - {p.sample_count} samples (enrolled: {enrolled_date})")
        print("-------------------------------------------\n")
        return 0

    if args.delete:
        pid = db._name_index.get(args.delete.lower().strip()) or args.delete.strip()
        success = db.delete_profile(args.delete)
        if success:
            context_db.delete_context(pid)
            print(f"Successfully deleted profile '{args.delete}' and erased all biometric and contextual data.")
            return 0
        else:
            print(f"Profile '{args.delete}' not found in database.")
            return 1

    if args.clear_all:
        confirm = input("Are you sure you want to delete ALL enrolled face profiles? (yes/no): ")
        if confirm.strip().lower() == "yes":
            db.clear_all()
            for p in db.list_profiles():
                context_db.delete_context(p.profile_id)
            print("All enrolled profiles and context have been erased.")
            return 0
        print("Aborted.")
        return 1

    if args.enroll:
        if not args.images:
            print("Error: --images <image_1> [image_2 ...] is required when using --enroll.")
            return 1
        try:
            profile = enroll_from_images(args.enroll, args.images, settings, db)
            if args.relationship:
                context_db.set_context(profile.profile_id, relationship=args.relationship, face_db=db)
            print(f"Successfully enrolled '{profile.display_name}' with {profile.sample_count} sample(s).")
            return 0
        except Exception as err:
            print(f"Enrollment failed: {err}")
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
