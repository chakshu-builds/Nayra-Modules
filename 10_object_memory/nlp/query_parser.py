"""NAYRA Module 10 — Dynamic Multilingual Query Parser for Object Memory.

Parses questions asked in English, Hindi, and Hinglish such as:
- 'Hey NAYRA, I asked about you where is my TV.'
- 'Or where is my laptop?'
- 'Where is the TV remote?'
- 'Where is my dining table?'
- 'Where are my keys?'
- 'What objects can you see?'
- 'Is my laptop visible?'
- 'Mera phone kahan hai?'
- 'Meri keys kahan rakhi thi?'
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Baseline COCO classes
DEFAULT_DETECTOR_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

@dataclass
class ParsedMemoryQuery:
    raw_query: str
    target_object: Optional[str]        # Canonical name (e.g. 'tv', 'phone', 'remote', 'bottle')
    display_name_hi: str
    display_name_en: str
    is_supported: bool                  # False for COCO out-of-vocabulary objects like keys, spectacles
    query_type: str                     # 'WHERE', 'LIST', 'CONFIRM_VISIBLE', 'UNKNOWN'
    detected_language: str              # 'hi', 'en', 'hinglish'
    is_follow_up: bool = False
    is_exact: bool = False


# Conversational lead-ins and filler patterns to strip
CONVERSATIONAL_FILLERS: List[re.Pattern] = [
    re.compile(r"^\s*(?:hey|hi|hello|ok|okay)?\s*\bnayra\b[\s,]*", re.IGNORECASE),
    re.compile(r"\b(?:hey|hi|hello|ok|okay)\s+nayra\b[\s,]*", re.IGNORECASE),
    re.compile(r"\bnayra\b[\s,]*", re.IGNORECASE),
    re.compile(
        r"^\s*(?:i asked about you|i asked you|i am asking about|i am asking you|"
        r"i want to ask you|i want to ask|i want to know|"
        r"can you tell me|could you tell me|would you tell me|tell me|please tell me|"
        r"can you find|could you find|please find|find|"
        r"can you locate|could you locate|please locate|locate|"
        r"can you see|do you see|have you seen|"
        r"do you know|do you remember|tell us|can you help me find)[\s,]+",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:or|and|so|well|actually|just|please|kindly)[\s,]+", re.IGNORECASE),
    re.compile(
        r"^\s*(?:acha|achha|suno|zara|kripya|mujhe batao|batao|"
        r"kya tum bata sakte ho|kya aap bata sakte ho|kya tumne dekha|kya aapne dekha)[\s,]+",
        re.IGNORECASE,
    ),
]

# High-frequency Alzheimer's / assistive items that are NOT in standard COCO 80
UNSUPPORTED_DEFINITIONS: List[Tuple[str, str, str, List[str]]] = [
    # (canonical, hi_name, en_name, aliases)
    ("glasses", "चश्मा", "eyeglasses", [
        "glasses", "eyeglasses", "spectacles", "specs", "reading glasses", "sunglasses",
        "chashma", "chashme", "chashmo", "चश्मा", "चश्मे"
    ]),
    ("keys", "चाबी", "keys", [
        "car keys", "house keys", "key", "keys", "chabhi", "chaabi", "chabi", "chhabiyan", "चाबी", "चाभियां"
    ]),
    ("wallet", "बटुआ", "wallet", [
        "wallet", "wallets", "money purse", "batua", "batuwa", "बटुआ"
    ]),
    ("door", "दरवाजा", "door", [
        "door", "doors", "front door", "room door", "darwaza", "दरवाजा"
    ]),
    ("stairs", "सीढ़ियां", "stairs", [
        "stairs", "staircase", "steps", "sidhiyan", "sidhiya", "सीढ़ियां"
    ]),
]

# Core personal and assistive object alias definitions with prioritized canonical names
CORE_SUPPORTED_DEFINITIONS: List[Tuple[str, str, str, List[str]]] = [
    # Remote control - MUST match before TV
    ("remote", "रिमोट", "remote", [
        "television remote", "tv remote", "remote control", "remotes", "remote", "remot", "रिमोट"
    ]),
    # TV / Television
    ("tv", "टीवी", "TV", [
        "smart tv", "television set", "television", "tv set", "t.v.", "t.v", "tv", "tvs", "televisions", "स्मार्ट टीवी", "टीवी"
    ]),
    # Cell phone
    ("phone", "फ़ोन", "phone", [
        "cell phone", "cellphone", "mobile phone", "smartphone", "smart phone",
        "cellphones", "mobiles", "phones", "phone", "mobile", "फ़ोन", "फोन", "मोबाइल"
    ]),
    # Laptop
    ("laptop", "लैपटॉप", "laptop", [
        "laptop computer", "macbook pro", "macbook air", "macbook", "notebook",
        "laptops", "laptop", "computer", "pc", "लैपटॉप", "कम्प्यूटर"
    ]),
    # Dining table / Table / Desk
    ("dining table", "डाइनिंग टेबल", "table", [
        "dining table", "dinner table", "coffee table", "study table", "work desk",
        "tables", "table", "desks", "desk", "mej", "mez", "डाइनिंग टेबल", "मेज"
    ]),
    # Couch / Sofa
    ("sofa", "सोफ़ा", "couch", [
        "living room sofa", "couch", "couches", "sofas", "sofa", "settee", "सोफा", "सोफ़ा"
    ]),
    # Bottle / Medicine
    ("bottle", "बोतल", "bottle", [
        "water bottle", "medicine bottle", "bottles", "bottle", "dawa", "dawai",
        "medicine", "capsule", "pill", "syrup", "बोतल", "दवा", "दवाई"
    ]),
    # Cup / Mug
    ("cup", "कप", "cup", [
        "coffee mug", "tea cup", "coffee cup", "mugs", "mug", "cups", "cup", "chai", "कप", "मग्ग"
    ]),
    # Book / Notebook
    ("book", "किताब", "book", [
        "books", "book", "diary", "notebook", "novel", "kitab", "pustak", "marybook", "किताब", "पुस्तक", "डायरी"
    ]),
    # Chair
    ("chair", "कुर्सी", "chair", [
        "office chair", "armchair", "chairs", "chair", "kursi", "kursiyaan", "कुर्सी"
    ]),
    # Bed
    ("bed", "बिस्तर", "bed", [
        "beds", "bed", "bistar", "palang", "mattress", "बिस्तर", "पलंग"
    ]),
    # Clock / Watch
    ("clock", "घड़ी", "clock", [
        "wrist watch", "wristwatch", "watches", "watch", "clocks", "clock", "wall clock", "ghadi", "ghari", "घड़ी", "घड़ी"
    ]),
    # Bag / Backpack
    ("bag", "बैग", "bag", [
        "backpack", "backpacks", "handbag", "handbags", "bags", "bag", "purse",
        "purses", "suitcase", "suitcases", "basta", "jhola", "बैग", "पर्स", "बस्ता", "झोला"
    ]),
    # Refrigerator
    ("fridge", "फ्रिज", "refrigerator", [
        "refrigerator", "refrigerators", "fridge", "fridges", "फ्रिज"
    ]),
    # Keyboard
    ("keyboard", "कीबोर्ड", "keyboard", [
        "keyboard", "keyboards", "कीबोर्ड"
    ]),
    # Mouse
    ("mouse", "माउस", "mouse", [
        "mouse", "mice", "माउस"
    ]),
    # Scissors
    ("scissors", "कैंची", "scissors", [
        "scissors", "scissor", "kainchi", "कैंची"
    ]),
    # Umbrella
    ("umbrella", "छाता", "umbrella", [
        "umbrella", "umbrellas", "chhatri", "chata", "छाता", "छतरी"
    ]),
    # Person
    ("person", "व्यक्ति", "person", [
        "person", "people", "man", "woman", "someone", "insan", "vyakti", "aadmi", "व्यक्ति", "इंसान", "आदमी"
    ]),
]

# Scene enumeration / List intent regexes
SCENE_LIST_PATTERNS = [
    r"\b(?:what can you see|what do you see|what objects can you see|what objects are around me)\b",
    r"\b(?:what is around me|what is in front of me|what's around me|what's in front of me)\b",
    r"\b(?:describe the scene|describe what you see|what do you remember|what objects do you remember)\b",
    r"\b(?:list objects|list items|list everything|list all objects|what all do you see)\b",
    r"\b(?:kya kya dikh raha hai|kya dekh sakte ho|aas paas kya hai|kya kya dekha|sab dikhao)\b",
]

# Confirmation / Visibility intent regexes
CONFIRM_VISIBLE_PATTERNS = [
    r"\b(?:is my|is the|are my|are the|can you see my|can you see the|do you see my|do you see the)\b",
    r"\b(?:kya .* dikh raha hai|kya .* dikh rahi hai|kya .* dekh sakte ho)\b",
]

# Where-intent regexes
WHERE_PATTERNS = [
    r"\b(?:kahan|kaha|kidhar|kdr|कहाँ|कहा|किधर)\b",
    r"\b(?:where|where's|where is|where are|where did|locate|find|last seen)\b",
    r"\b(?:dekha tha|rakha tha|rakhi thi|chhoda tha|choda)\b",
    r"\b(?:kept|placed|left|rakha|rakhe)\b",
]

# Follow-up pronoun intent regexes ('Where was it?', 'Where did you see it?')
FOLLOW_UP_PATTERNS = [
    r"\b(?:where was it|where is it|where was it kept|where did you see it|where did i leave it|where was that|where is that|where was this|where is this|where was it placed|where were they|where were they kept|where is it now)\b",
    r"\b(?:kahan tha wo|wo kahan tha|kahan rakha tha usse|kahan dekha tha usse|kahan hai wo|wo kahan hai)\b",
    r"\b(?:where was it exactly|where exactly was it|where was that exactly)\b",
]

# Exact location request regexes ('Where was it exactly?')
EXACT_PATTERNS = [
    r"\b(?:exact|exactly|exact location|precise location|precisely|theek kahan|kahan tha theek)\b",
]


class MemoryQueryParser:
    """Dynamically parses natural language location and scene queries."""

    def __init__(self, detector_classes: Optional[Sequence[str]] = None) -> None:
        self.detector_classes: List[str] = list(detector_classes or DEFAULT_DETECTOR_CLASSES)
        self._compiled_where = [re.compile(p, re.IGNORECASE) for p in WHERE_PATTERNS]
        self._compiled_list = [re.compile(p, re.IGNORECASE) for p in SCENE_LIST_PATTERNS]
        self._compiled_confirm = [re.compile(p, re.IGNORECASE) for p in CONFIRM_VISIBLE_PATTERNS]
        self._compiled_follow_up = [re.compile(p, re.IGNORECASE) for p in FOLLOW_UP_PATTERNS]
        self._compiled_exact = [re.compile(p, re.IGNORECASE) for p in EXACT_PATTERNS]

        self._build_alias_matchers()

    def update_vocabulary(self, detector_classes: Sequence[str]) -> None:
        """Dynamically update supported vocabulary from active detector classes."""
        self.detector_classes = list(detector_classes)
        self._build_alias_matchers()

    def _build_alias_matchers(self) -> None:
        """Compile alias patterns sorted strictly by length descending for specificity."""
        # Entries: (compiled_pattern, canonical, hi_name, en_name, is_supported)
        entries: List[Tuple[re.Pattern, str, str, str, bool]] = []
        seen_aliases: Set[str] = set()

        # 1. Unsupported objects (glasses, keys, wallet, door, stairs)
        for canon, hi_name, en_name, aliases in UNSUPPORTED_DEFINITIONS:
            for alias in aliases:
                norm_alias = alias.lower().strip()
                if norm_alias not in seen_aliases:
                    seen_aliases.add(norm_alias)
                    # Use word boundary; escape for regex
                    pat = re.compile(rf"\b{re.escape(norm_alias)}\b", re.IGNORECASE)
                    entries.append((pat, canon, hi_name, en_name, False))

        # 2. Core supported objects
        for canon, hi_name, en_name, aliases in CORE_SUPPORTED_DEFINITIONS:
            for alias in aliases:
                norm_alias = alias.lower().strip()
                if norm_alias not in seen_aliases:
                    seen_aliases.add(norm_alias)
                    pat = re.compile(rf"\b{re.escape(norm_alias)}\b", re.IGNORECASE)
                    entries.append((pat, canon, hi_name, en_name, True))

        # 3. Dynamic detector classes not covered in core definitions
        for det_cls in self.detector_classes:
            raw_clean = det_cls.lower().strip()
            # Standardize e.g. "cell phone" -> "phone"
            canon = raw_clean
            if canon in seen_aliases:
                continue
            seen_aliases.add(canon)
            pat = re.compile(rf"\b{re.escape(canon)}\b", re.IGNORECASE)
            entries.append((pat, canon, canon, canon, True))

            # Also add plural form if not already present
            plural = canon + "s"
            if plural not in seen_aliases:
                seen_aliases.add(plural)
                pat_pl = re.compile(rf"\b{re.escape(plural)}\b", re.IGNORECASE)
                entries.append((pat_pl, canon, canon, canon, True))

        # CRITICAL: Sort by alias pattern length in DESCENDING order so longer/compound
        # phrases (e.g. "tv remote") match BEFORE individual sub-words (e.g. "tv")
        entries.sort(key=lambda item: len(item[0].pattern), reverse=True)
        self._compiled_matchers = entries

    @classmethod
    def strip_conversational_fillers(cls, text: str) -> str:
        """Strip conversational lead-ins, wake words, and filler conjunctions."""
        cleaned = text.strip()
        changed = True
        while changed and cleaned:
            changed = False
            for pat in CONVERSATIONAL_FILLERS:
                new_text = pat.sub("", cleaned)
                if new_text != cleaned:
                    cleaned = new_text.strip()
                    changed = True
                    break
        return cleaned

    def parse(self, text: str) -> ParsedMemoryQuery:
        """Parse natural language query into a structured ParsedMemoryQuery."""
        raw = text.strip()
        if not raw:
            return ParsedMemoryQuery(
                raw_query="",
                target_object=None,
                display_name_hi="",
                display_name_en="",
                is_supported=False,
                query_type="UNKNOWN",
                detected_language="en",
            )

        lang = self._detect_language(raw)

        # 1. Clean conversational lead-ins and wake words
        cleaned = self.strip_conversational_fillers(raw)
        if not cleaned:
            cleaned = raw

        # 2. Check for scene enumeration / LIST intent (e.g. 'What can you see?')
        is_list_query = any(p.search(cleaned) or p.search(raw) for p in self._compiled_list)

        # 3. Check for specific target object (using specificity-ordered matchers)
        matched_target: Optional[str] = None
        matched_hi: str = ""
        matched_en: str = ""
        is_supported: bool = False

        for pat, canon, hi_name, en_name, supported in self._compiled_matchers:
            if pat.search(cleaned) or pat.search(raw):
                matched_target = canon
                matched_hi = hi_name
                matched_en = en_name
                is_supported = supported
                break

        # If LIST intent matched and NO specific object target was requested
        if is_list_query and matched_target is None:
            return ParsedMemoryQuery(
                raw_query=raw,
                target_object=None,
                display_name_hi="",
                display_name_en="",
                is_supported=True,
                query_type="LIST",
                detected_language=lang,
            )

        # 4. Determine query intent and follow-up status
        is_confirm_query = any(p.search(cleaned) or p.search(raw) for p in self._compiled_confirm)
        is_where_query = any(p.search(cleaned) or p.search(raw) for p in self._compiled_where)
        is_follow_up = any(p.search(cleaned) or p.search(raw) for p in self._compiled_follow_up)
        is_exact = any(p.search(cleaned) or p.search(raw) for p in self._compiled_exact)

        query_type = "UNKNOWN"
        if is_list_query:
            query_type = "LIST"
        elif is_where_query or (is_follow_up and matched_target is None):
            query_type = "WHERE"
        elif is_confirm_query and matched_target is not None:
            query_type = "CONFIRM_VISIBLE"
        elif matched_target is not None:
            query_type = "WHERE"

        if matched_target is None and (is_follow_up or (is_exact and is_where_query)):
            is_follow_up = True
            query_type = "WHERE"

        return ParsedMemoryQuery(
            raw_query=raw,
            target_object=matched_target,
            display_name_hi=matched_hi,
            display_name_en=matched_en,
            is_supported=is_supported,
            query_type=query_type,
            detected_language=lang,
            is_follow_up=is_follow_up,
            is_exact=is_exact,
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect whether query is Hindi, Hinglish, or English."""
        # Check Devanagari Unicode block
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
        if has_devanagari:
            return "hi"

        # Check Hinglish keywords
        hinglish_words = {
            "mera", "meri", "mere", "kahan", "kaha", "hai", "rakha", "rakhi",
            "thi", "tha", "chhoda", "maine", "dikh", "raha", "rahi", "kidhar"
        }
        lower_tokens = set(re.findall(r"\b\w+\b", text.lower()))
        if any(w in lower_tokens for w in hinglish_words):
            return "hinglish"

        return "en"
