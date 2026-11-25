"""CLI management tool for NAYRA Personal Context Database."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add current package directory to sys.path
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from config.settings import get_default_person_recognition_settings
from context_database import PersonContextDatabase
from face_database import FaceDatabase

logger = logging.getLogger(__name__)


def resolve_profile_id(identifier: str, face_db: FaceDatabase) -> str | None:
    """Resolve identifier as either an exact profile_id or display name."""
    clean = identifier.strip()
    profile = face_db.get_profile(clean)
    if profile:
        return profile.profile_id
    by_name = face_db.get_profile_by_name(clean)
    if by_name:
        return by_name.profile_id
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="NAYRA - Personal Context Database Manager")
    parser.add_argument("--list", action="store_true", help="List all enrolled profiles with their personal context.")
    parser.add_argument("--show", type=str, default=None, help="Show context for a person by name or profile_id.")
    parser.add_argument("--set", type=str, default=None, metavar="NAME_OR_ID", help="Create or update context for a person.")
    parser.add_argument("--relationship", type=str, default=None, help="Relationship (e.g. Sister, Friend, Caregiver).")
    parser.add_argument("--occupation", type=str, default=None, help="Occupation or profession.")
    parser.add_argument("--description", type=str, default=None, help="Personal description or visual/behavioral cues.")
    parser.add_argument("--notes", type=str, default=None, help="General user-provided notes.")
    parser.add_argument("--add-fact", type=str, default=None, help="Add a specific personal fact.")
    parser.add_argument("--add-pref", type=str, default=None, help="Add a personal preference.")
    parser.add_argument("--category", type=str, default="general", help="Category for fact or preference (default: general).")
    parser.add_argument("--remove-fact", type=int, default=None, help="Remove a fact by ID.")
    parser.add_argument("--remove-pref", type=int, default=None, help="Remove a preference by ID.")
    parser.add_argument("--delete", type=str, default=None, help="Delete all context for a person by name or profile_id.")
    parser.add_argument("--sync", action="store_true", help="Prune context records for profiles deleted from FaceDatabase.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    settings = get_default_person_recognition_settings()
    face_db = FaceDatabase(settings)
    context_db = PersonContextDatabase(settings)

    if args.sync:
        pruned = context_db.sync_with_face_db(face_db)
        print(f"Sync complete. Pruned {pruned} orphaned context record(s).")
        return 0

    if args.remove_fact is not None:
        if context_db.remove_fact(args.remove_fact):
            print(f"Fact #{args.remove_fact} removed successfully.")
            return 0
        else:
            print(f"Fact #{args.remove_fact} not found.")
            return 1

    if args.remove_pref is not None:
        if context_db.remove_preference(args.remove_pref):
            print(f"Preference #{args.remove_pref} removed successfully.")
            return 0
        else:
            print(f"Preference #{args.remove_pref} not found.")
            return 1

    if args.delete:
        pid = resolve_profile_id(args.delete, face_db)
        target_id = pid or args.delete.strip()
        if context_db.delete_context(target_id):
            print(f"Personal context for '{args.delete}' deleted successfully.")
            return 0
        else:
            print(f"No context record found for '{args.delete}'.")
            return 1

    if args.set:
        pid = resolve_profile_id(args.set, face_db)
        if not pid:
            print(f"Error: Profile '{args.set}' not found in FaceDatabase. Context can only link to enrolled profiles.")
            return 1
        ctx = context_db.update_profile_details(
            profile_id=pid,
            relationship=args.relationship,
            occupation=args.occupation,
            description=args.description,
            notes=args.notes,
            face_db=face_db,
        )
        print(f"Updated context for '{args.set}' (ID: {pid[:8]}):")
        print(f" • Relationship: {ctx.relationship or '(none)'}")
        print(f" • Occupation:   {ctx.occupation or '(none)'}")
        print(f" • Description:  {ctx.description or '(none)'}")
        print(f" • Notes:        {ctx.notes or '(none)'}")
        return 0

    if args.add_fact:
        if not args.set and not args.show:
            print("Error: Specify person with --set <name> or --show <name> when adding a fact.")
            return 1
        target = args.set or args.show
        pid = resolve_profile_id(target, face_db)
        if not pid:
            print(f"Error: Profile '{target}' not found in FaceDatabase.")
            return 1
        fact_item = context_db.add_fact(pid, args.add_fact, category=args.category, face_db=face_db)
        print(f"Added Fact #{fact_item.id} to '{target}': \"{fact_item.fact}\" [{fact_item.category}]")
        return 0

    if args.add_pref:
        if not args.set and not args.show:
            print("Error: Specify person with --set <name> or --show <name> when adding a preference.")
            return 1
        target = args.set or args.show
        pid = resolve_profile_id(target, face_db)
        if not pid:
            print(f"Error: Profile '{target}' not found in FaceDatabase.")
            return 1
        pref_item = context_db.add_preference(pid, args.add_pref, category=args.category, face_db=face_db)
        print(f"Added Preference #{pref_item.id} to '{target}': \"{pref_item.preference}\" [{pref_item.category}]")
        return 0

    if args.show:
        pid = resolve_profile_id(args.show, face_db)
        if not pid:
            print(f"Profile '{args.show}' not found in FaceDatabase.")
            return 1
        profile = face_db.get_profile(pid)
        name = profile.display_name if profile else args.show
        ctx = context_db.get_context(pid, face_db=face_db)
        print(f"\n--- Personal Context: {name} [{pid[:8]}] ---")
        if not ctx:
            print("No personal context recorded for this profile.")
            return 0
        print(f"Relationship: {ctx.relationship or '(none)'}")
        print(f"Occupation:   {ctx.occupation or '(none)'}")
        print(f"Description:  {ctx.description or '(none)'}")
        print(f"Notes:        {ctx.notes or '(none)'}")
        print("\nImportant Facts:")
        if not ctx.facts:
            print("  (no facts recorded)")
        for f in ctx.facts:
            print(f"  • [#{f.id}] ({f.category}) {f.fact}")
        print("\nPreferences:")
        if not ctx.preferences:
            print("  (no preferences recorded)")
        for p in ctx.preferences:
            print(f"  • [#{p.id}] ({p.category}) {p.preference}")
        return 0

    if args.list or len(sys.argv) == 1:
        profiles = face_db.list_profiles()
        print(f"\n--- Enrolled People & Personal Context ({len(profiles)}) ---")
        if not profiles:
            print("No enrolled profiles found.")
            return 0
        for p in profiles:
            ctx = context_db.get_context(p.profile_id)
            rel_str = f" ({ctx.relationship})" if ctx and ctx.relationship else ""
            print(f"\n• [{p.profile_id[:8]}] {p.display_name}{rel_str}")
            if ctx:
                if ctx.occupation:
                    print(f"    Occupation:  {ctx.occupation}")
                if ctx.description:
                    print(f"    Description: {ctx.description}")
                if ctx.notes:
                    print(f"    Notes:       {ctx.notes}")
                if ctx.facts:
                    print(f"    Facts ({len(ctx.facts)}):")
                    for f in ctx.facts:
                        print(f"      - [#{f.id}] ({f.category}) {f.fact}")
                if ctx.preferences:
                    print(f"    Preferences ({len(ctx.preferences)}):")
                    for pref in ctx.preferences:
                        print(f"      - [#{pref.id}] ({pref.category}) {pref.preference}")
            else:
                print("    (No contextual details recorded yet)")
        print("\n--------------------------------------------------------------")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
