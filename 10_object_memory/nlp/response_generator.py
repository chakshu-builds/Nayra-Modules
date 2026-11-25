"""NAYRA Module 10 — Patient-Friendly Empathetic Response Generator.

Generates reassuring, clear spoken responses in Hindi, English, and Hinglish.
Differentiates between:
1. Currently visible objects in live camera view (actionable immediate guidance with left/center/right and furniture anchors).
2. Historical observations when camera is active (clearly stating it was seen previously and is not currently visible).
3. Standard memory recall when camera is offline (stating time elapsed and location).
4. Unsupported objects (empathetic transparency).
5. Unobserved objects.
"""

from __future__ import annotations

from datetime import datetime
import time
from typing import Optional

try:
    from ..db.models import MemoryQueryResult, ObjectObservation
    from ..perception.spatial_context import ANCHOR_HINDI_NAMES
    from .query_parser import ParsedMemoryQuery
except (ImportError, ValueError):
    from db.models import MemoryQueryResult, ObjectObservation
    from perception.spatial_context import ANCHOR_HINDI_NAMES
    from nlp.query_parser import ParsedMemoryQuery

FEMININE_OBJECTS = {"bottle", "book", "umbrella", "clock", "scissors", "bag"}


class ResponseGenerator:
    """Creates reassuring, accurate spoken answers based on object location observations."""

    @staticmethod
    def generate_unknown_followup_response(query: ParsedMemoryQuery) -> MemoryQueryResult:
        """Response when a follow-up query with pronouns is asked without prior context."""
        msg_hi = (
            "Mujhe samajh nahi aaya ki aap kis cheez ke baare mein pooch rahe hain. "
            "Kripya poora bolein, jaise 'Mera phone kahan hai?' ya 'Where is my book?'."
        )
        msg_en = (
            "I'm not sure which object you mean. "
            "Please tell me what you are looking for, for example 'Where is my book?'."
        )
        return MemoryQueryResult(
            found=False,
            canonical_name="unknown",
            observation=None,
            spoken_response_hi=msg_hi,
            spoken_response_en=msg_en,
            confidence_level="NONE",
            time_category="OLD",
            is_currently_visible=False,
        )

    @staticmethod
    def generate_response(
        query: ParsedMemoryQuery,
        obs: Optional[ObjectObservation],
        is_currently_visible: Optional[bool] = None,
    ) -> MemoryQueryResult:
        """Formulate bilingual responses for a query and an observation.

        Args:
            query: Parsed query structure.
            obs: Retrieved observation (live or from SQLite).
            is_currently_visible:
                True  -> Object is actively seen in live camera right now.
                False -> Live camera was checked, object is not currently in view.
                None  -> Camera offline / pure historical database query.
        """
        canonical = query.target_object or "object"

        # Case -1: List / Scene Enumeration query (e.g. 'What can you see?')
        if query.query_type == "LIST":
            return ResponseGenerator.generate_scene_list_response(query, [])

        # Case 0: No target object identified (garbled query or empty target)
        if not query.target_object:
            if getattr(query, "is_follow_up", False):
                return ResponseGenerator.generate_unknown_followup_response(query)
            msg_hi = (
                "Mujhe samajh nahi aaya ki aap kaun si cheez dhoondh rahe hain. "
                "Kripya doobara bolein, jaise 'Mera phone kahan hai?' ya 'Where is my laptop?'."
            )
            msg_en = (
                "I could not identify the object you are looking for. "
                "Please ask again, for example 'Where is my phone?' or 'Where is my laptop?'."
            )
            return MemoryQueryResult(
                found=False,
                canonical_name="unknown",
                observation=None,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="NONE",
                time_category="OLD",
                is_currently_visible=False,
            )

        # STRICT GROUNDING: Discard any mismatched observation
        if obs is not None and obs.canonical_name != canonical:
            obs = None

        # Case 1: Unsupported object (e.g. glasses, keys)
        if not query.is_supported and query.target_object:
            hi_name = query.display_name_hi or canonical
            en_name = query.display_name_en or canonical
            msg_hi = (
                f"Main {hi_name} ko pehchanne mein abhi saksham nahi hoon. "
                f"Kripya kisi parivar ke sadasya ya dekhbhal karne wale ki madad lein."
            )
            msg_en = (
                f"I cannot currently recognize {en_name}. "
                f"Please ask a family member or caregiver to help you locate it."
            )
            return MemoryQueryResult(
                found=False,
                canonical_name=canonical,
                observation=None,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="NONE",
                time_category="OLD",
                is_currently_visible=False,
            )

        # Case 2: No observation found in memory
        if obs is None:
            hi_name = query.display_name_hi or canonical
            en_name = query.display_name_en or canonical
            if is_currently_visible is False:
                msg_hi = (
                    f"Mujhe aapka {hi_name} abhi tak kahin dikhai nahi diya. "
                    f"Abhi camera mein bhi nahi dikh raha."
                )
                msg_en = (
                    f"I haven't seen your {en_name} yet, "
                    f"and it is not currently visible in the camera."
                )
            else:
                msg_hi = (
                    f"Mujhe aapka {hi_name} abhi tak kahin dikhai nahi diya. "
                    f"Jaise hi camera ke saamne aayega, main yaad rakhunga."
                )
                msg_en = (
                    f"I haven't seen your {en_name} yet. "
                    f"As soon as it appears in front of the camera, I will remember its location."
                )
            return MemoryQueryResult(
                found=False,
                canonical_name=canonical,
                observation=None,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="NONE",
                time_category="OLD",
                is_currently_visible=False,
            )

        hi_name = obs.hindi_alias or query.display_name_hi or canonical
        en_name = query.display_name_en or canonical
        is_fem = canonical in FEMININE_OBJECTS
        dikh_str = "dikh rahi hai" if is_fem else "dikh raha hai"

        anc_raw = (obs.anchor_object or "").lower().strip()
        anc_hi = ANCHOR_HINDI_NAMES.get(anc_raw, anc_raw) if anc_raw else None
        if " near the " in anc_raw:
            parts = anc_raw.split(" near the ")
            p0 = parts[0].strip()
            p1 = parts[1].strip()
            hi_p0 = "table" if p0 in ("dining table", "table") else ANCHOR_HINDI_NAMES.get(p0, p0)
            hi_p1 = ANCHOR_HINDI_NAMES.get(p1, p1)
            anc_spoken_hi = f"{hi_p1} ke paas, {hi_p0}"
            anc_en = f"{p0} near the {p1}"
            anc_hi = anc_spoken_hi
        else:
            anc_spoken_hi = "table" if anc_hi in ("dining table", "table") else anc_hi
            anc_en = anc_raw if anc_raw else None
        rel = (obs.spatial_relation or "on").lower()

        # -------------------------------------------------------------
        # Mode A: CURRENTLY VISIBLE IN LIVE CAMERA VIEW
        # -------------------------------------------------------------
        if is_currently_visible is True:
            zone = (obs.horizontal_zone or "CENTER").upper()

            if anc_spoken_hi:
                if rel == "on":
                    if zone == "CENTER":
                        msg_hi = f"Aapka {hi_name} abhi saamne {anc_spoken_hi} par hai."
                        msg_en = f"Your {en_name} is on the {anc_en} right in front of you."
                    elif zone == "LEFT":
                        msg_hi = f"{hi_name.capitalize()} {anc_spoken_hi} par, aapki left side mein hai."
                        msg_en = f"Your {en_name} is on the {anc_en}, on your left side."
                    elif zone == "RIGHT":
                        msg_hi = f"{hi_name.capitalize()} {anc_spoken_hi} par, aapke right side mein hai."
                        msg_en = f"Your {en_name} is on the {anc_en}, on your right side."
                    else:
                        msg_hi = f"Aapka {hi_name} {anc_spoken_hi} par {dikh_str}."
                        msg_en = f"Your {en_name} is visible on the {anc_en}."
                else:  # near
                    if zone == "LEFT":
                        msg_hi = f"{hi_name.capitalize()} {anc_spoken_hi} ke paas, aapki left side mein hai."
                        msg_en = f"The {en_name} is near the {anc_en}, on your left side."
                    elif zone == "RIGHT":
                        msg_hi = f"{hi_name.capitalize()} {anc_spoken_hi} ke paas, aapke right side mein hai."
                        msg_en = f"The {en_name} is near the {anc_en}, on your right side."
                    else:  # CENTER
                        msg_hi = f"{hi_name.capitalize()} saamne {anc_spoken_hi} ke paas hai."
                        msg_en = f"The {en_name} is near the {anc_en}, right in front of you."
            else:
                if zone == "LEFT":
                    msg_hi = f"{hi_name.capitalize()} aapki left side mein {dikh_str}."
                    msg_en = f"The {en_name} is visible on your left side."
                elif zone == "RIGHT":
                    msg_hi = f"{hi_name.capitalize()} aapke right side mein {dikh_str}."
                    msg_en = f"The {en_name} is visible on your right side."
                else:
                    msg_hi = f"{hi_name.capitalize()} aapke saamne {dikh_str}."
                    msg_en = f"The {en_name} is visible right in front of you."

            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="HIGH",
                time_category="NOW",
                is_currently_visible=True,
            )

        # -------------------------------------------------------------
        # Mode B: NOT CURRENTLY VISIBLE (Live camera checked, absent)
        # -------------------------------------------------------------
        if is_currently_visible is False:
            elapsed_sec = int(round(obs.elapsed_seconds))
            elapsed_min = int(round(obs.elapsed_minutes))
            if elapsed_sec < 60:
                time_hi = f"lagbhag {elapsed_sec} second pehle"
                time_en = f"around {elapsed_sec} seconds ago"
                cat = "RECENT"
            elif elapsed_min <= 15:
                time_hi = "kuch der pehle"
                time_en = "a little while ago"
                cat = "RECENT"
            elif elapsed_min <= 120:
                hours = int(round(elapsed_min / 60.0))
                time_hi = f"lagbhag {hours} ghante pehle" if hours >= 1 else f"lagbhag {elapsed_min} minute pehle"
                time_en = f"about {hours} hour{'s' if hours > 1 else ''} ago" if hours >= 1 else f"about {elapsed_min} minutes ago"
                cat = "EARLIER"
            else:
                hours = int(round(elapsed_min / 60.0))
                time_hi = f"lagbhag {hours} ghante pehle"
                time_en = f"about {hours} hours ago"
                cat = "OLD"

            zone = (obs.horizontal_zone or "").upper()
            zone_en_phrase = (
                f"on your {zone.lower()} side" if zone in ("LEFT", "RIGHT")
                else ("right in front of you" if zone == "CENTER" else None)
            )
            zone_hi_phrase = (
                f"aapki {zone.lower()} side mein" if zone in ("LEFT", "RIGHT")
                else ("saamne" if zone == "CENTER" else None)
            )

            # Check if exact coordinates/position follow-up requested
            if getattr(query, "is_exact", False):
                details_en = []
                details_hi = []
                if zone_en_phrase:
                    details_en.append(zone_en_phrase)
                    details_hi.append(zone_hi_phrase)
                if obs.vertical_zone:
                    details_en.append(f"in the {obs.vertical_zone.lower()} area of the view")
                    details_hi.append(f"{obs.vertical_zone.lower()} hisse mein")
                if anc_en:
                    rel_word_en = "on" if obs.spatial_relation == "on" else "near"
                    rel_word_hi = "par" if obs.spatial_relation == "on" else "ke paas"
                    details_en.append(f"{rel_word_en} the {anc_en}")
                    details_hi.append(f"{anc_spoken_hi} {rel_word_hi}")

                loc_desc_en = ", ".join(details_en) if details_en else "in the camera view"
                loc_desc_hi = ", ".join(details_hi) if details_hi else "camera ke saamne"
                msg_en = (
                    f"I last saw your {en_name} {loc_desc_en}, {time_en}. "
                    f"It is not currently visible in the camera."
                )
                msg_hi = (
                    f"Maine aapka {hi_name} {loc_desc_hi}, {time_hi} dekha tha. "
                    f"Abhi camera mein nahi dikh raha."
                )
            elif anc_spoken_hi and zone_en_phrase:
                rel_word_en = "on" if obs.spatial_relation == "on" else "near"
                rel_word_hi = "par" if obs.spatial_relation == "on" else "ke paas"
                msg_hi = (
                    f"Maine aapka {hi_name} aakhri baar {anc_spoken_hi} {rel_word_hi}, {zone_hi_phrase}, {time_hi} dekha tha. "
                    f"Abhi camera mein nahi dikh raha."
                )
                msg_en = (
                    f"I saw your {en_name} {rel_word_en} the {anc_en}, {zone_en_phrase}, {time_en}. "
                    f"It is not currently visible in the camera."
                )
            elif anc_spoken_hi:
                rel_word_en = "on" if obs.spatial_relation == "on" else "near"
                rel_word_hi = "par" if obs.spatial_relation == "on" else "ke paas"
                msg_hi = (
                    f"Maine aapka {hi_name} {anc_spoken_hi} {rel_word_hi} {time_hi} dekha tha. "
                    f"Abhi camera mein nahi dikh raha."
                )
                msg_en = (
                    f"I saw your {en_name} {rel_word_en} the {anc_en}, {time_en}. "
                    f"It is not currently visible in the camera."
                )
            elif zone_en_phrase:
                msg_hi = (
                    f"Maine aapka {hi_name} aakhri baar {zone_hi_phrase}, {time_hi} dekha tha. "
                    f"Abhi camera mein nahi dikh raha."
                )
                msg_en = (
                    f"I last saw your {en_name} {zone_en_phrase}, {time_en}. "
                    f"It is not currently visible in the camera."
                )
            else:
                msg_hi = (
                    f"Maine aapka {hi_name} {time_hi} dekha tha, lekin uski sahi jagah record nahi ho paayi thi. "
                    f"Abhi camera mein nahi dikh raha."
                )
                msg_en = (
                    f"I saw your {en_name} {time_en}, but I didn't record its exact position. "
                    f"It is not currently visible in the camera."
                )

            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="HIGH" if cat == "RECENT" else "UNCERTAIN",
                time_category=cat,
                is_currently_visible=False,
            )

        # -------------------------------------------------------------
        # Mode C: STANDARD MEMORY RECALL (is_currently_visible is None)
        # -------------------------------------------------------------
        elapsed_sec = obs.elapsed_seconds
        elapsed_min = int(round(obs.elapsed_minutes))
        obs_time = datetime.fromtimestamp(obs.timestamp)
        time_str = obs_time.strftime("%I:%M").lstrip("0")
        am_pm = obs_time.strftime("%p")

        if obs.spatial_relation and obs.anchor_object:
            anc = anc_en or obs.anchor_object
            if obs.spatial_relation == "on":
                loc_en = f"on the {anc}"
                loc_hi = f"{anc_hi or anc_spoken_hi or anc} par"
            else:
                loc_en = f"near the {anc}"
                loc_hi = f"{anc_hi or anc_spoken_hi or anc} ke paas"
        elif obs.horizontal_zone:
            if obs.horizontal_zone == "LEFT":
                loc_en = "on your left"
                loc_hi = "baayein taraf"
            elif obs.horizontal_zone == "RIGHT":
                loc_en = "on your right"
                loc_hi = "daayein taraf"
            else:
                loc_en = "right in front of you"
                loc_hi = "saamne"
        else:
            loc_en = "in the room"
            loc_hi = "kamre mein"

        if obs.room_label:
            loc_en += f" in the {obs.room_label}"
            loc_hi += f", {obs.room_label} mein"

        # Recent (< 15s)
        if elapsed_sec < 15.0:
            msg_hi = f"Aapka {hi_name} abhi aapke {loc_hi} dikh raha hai."
            msg_en = f"Your {en_name} is currently visible {loc_en}."
            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="HIGH",
                time_category="NOW",
                is_currently_visible=True,
            )
        # Recent (< 15m)
        elif elapsed_min <= 15:
            min_str_hi = "kuch hi minute" if elapsed_min <= 2 else f"lagbhag {elapsed_min} minute"
            msg_hi = (
                f"Maine aapka {hi_name} {min_str_hi} pehle, {time_str} baje {loc_hi} dekha tha. "
                f"Yeh wahin hona chahiye."
            )
            min_str_en = "a couple of minutes" if elapsed_min <= 2 else f"about {elapsed_min} minutes"
            msg_en = (
                f"I saw your {en_name} {min_str_en} ago, at {time_str} {am_pm} {loc_en}. "
                f"It should still be there."
            )
            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="HIGH",
                time_category="RECENT",
                is_currently_visible=False,
            )
        # Earlier (15m to 2h)
        elif elapsed_min <= 120:
            hours = int(round(elapsed_min / 60.0))
            dur_hi = f"lagbhag {hours} ghante" if hours >= 1 else f"lagbhag {elapsed_min} minute"
            dur_en = f"about {hours} hour{'s' if hours > 1 else ''}" if hours >= 1 else f"about {elapsed_min} minutes"

            msg_hi = (
                f"Maine aapka {hi_name} {dur_hi} pehle {time_str} baje {loc_hi} dekha tha. "
                f"Ho sakta hai kisi ne hata diya ho."
            )
            msg_en = (
                f"I saw your {en_name} {dur_en} ago at {time_str} {am_pm} {loc_en}. "
                f"It might have been moved."
            )
            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="UNCERTAIN",
                time_category="EARLIER",
                is_currently_visible=False,
            )
        # Old (> 2h)
        else:
            hours = int(round(elapsed_min / 60.0))
            msg_hi = (
                f"Maine aapka {hi_name} kafi pehle, lagbhag {hours} ghante pehle {loc_hi} dekha tha. "
                f"Kripya wahan check karein ya kisi ki madad lein."
            )
            msg_en = (
                f"I last saw your {en_name} quite a while ago, about {hours} hours ago {loc_en}. "
                f"Please check there or ask someone to assist you."
            )
            return MemoryQueryResult(
                found=True,
                canonical_name=canonical,
                observation=obs,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="UNCERTAIN",
                time_category="OLD",
                is_currently_visible=False,
            )

    @staticmethod
    def generate_scene_list_response(
        query: ParsedMemoryQuery,
        visible_objects: Sequence[str],
    ) -> MemoryQueryResult:
        """Formulate spoken response enumerating all currently detected objects."""
        if not visible_objects:
            msg_en = "I cannot see any recognized objects in front of the camera right now."
            msg_hi = "Abhi camera ke saamne koi pehchani hui cheez nahi dikh rahi hai."
            return MemoryQueryResult(
                found=False,
                canonical_name="all",
                observation=None,
                spoken_response_hi=msg_hi,
                spoken_response_en=msg_en,
                confidence_level="NONE",
                time_category="NOW",
                is_currently_visible=False,
            )

        unique_objs = sorted(list(dict.fromkeys(visible_objects)))
        if len(unique_objs) == 1:
            en_items = f"a {unique_objs[0]}"
            hi_items = unique_objs[0]
        elif len(unique_objs) == 2:
            en_items = f"a {unique_objs[0]} and a {unique_objs[1]}"
            hi_items = f"{unique_objs[0]} aur {unique_objs[1]}"
        else:
            en_items = ", ".join(f"a {o}" for o in unique_objs[:-1]) + f", and a {unique_objs[-1]}"
            hi_items = ", ".join(unique_objs[:-1]) + f", aur {unique_objs[-1]}"

        msg_en = f"Right now, I can see {en_items} in front of the camera."
        msg_hi = f"Abhi camera ke saamne mujhe {hi_items} dikh raha hai."

        return MemoryQueryResult(
            found=True,
            canonical_name="all",
            observation=None,
            spoken_response_hi=msg_hi,
            spoken_response_en=msg_en,
            confidence_level="HIGH",
            time_category="NOW",
            is_currently_visible=True,
        )
