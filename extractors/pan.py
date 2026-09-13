"""
extractors/pan.py
Extractor for Indian PAN Cards (Permanent Account Number).

Public API:
    extract(rec_texts, rec_scores=None)      -> (canonical: dict, confidence: dict)
    extract_from_markdown(md_text: str)      -> (canonical: dict, confidence: dict)

extract_from_markdown() is the preferred path when PaddleOCRVL is used as the
OCR backend, since the structured markdown output preserves label→value adjacency
much better than the flat token stream used by extract().
"""
import re
from datetime import datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_at(scores, idx):
    try:
        return round(float(scores[idx]), 4)
    except (IndexError, TypeError, ValueError):
        return None


def _to_iso_date(date_str: str) -> str:
    """Convert DD/MM/YYYY or DD-MM-YYYY to ISO 8601."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


# PAN number: 5 uppercase letters, 4 digits, 1 uppercase letter
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Date pattern: DD/MM/YYYY or DD-MM-YYYY
_DATE_RE = re.compile(r"\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b")


def _extract_raw(rec_texts, rec_scores):
    data, scores = {}, {}

    for i, text in enumerate(rec_texts):
        t = text.strip()

        # PAN number — can appear standalone or inline
        if _PAN_RE.match(t) and "PAN" not in data:
            data["PAN"] = t.upper()
            scores["PAN"] = _score_at(rec_scores, i)

        elif "Permanent Account Number" in t or "Income Tax Department" in t:
            # PAN may be on the next line
            if i + 1 < len(rec_texts):
                next_t = rec_texts[i + 1].strip()
                m = _PAN_RE.search(next_t)
                if m and "PAN" not in data:
                    data["PAN"] = m.group(0).upper()
                    scores["PAN"] = _score_at(rec_scores, i + 1)

        # Name — appears after label "Name" on PAN card
        elif t.upper() == "NAME" or t == "Name":
            if i + 1 < len(rec_texts) and "Name" not in data:
                data["Name"] = rec_texts[i + 1].strip()
                scores["Name"] = _score_at(rec_scores, i + 1)

        # Father's name — label must contain both "Father" and "Name"
        elif "Father" in t and "Name" in t:
            if i + 1 < len(rec_texts) and "Father's Name" not in data:
                data["Father's Name"] = rec_texts[i + 1].strip()
                scores["Father's Name"] = _score_at(rec_scores, i + 1)

        # Date of Birth
        elif "Date of Birth" in t or t.upper() in ("DATE OF BIRTH", "DOB"):
            if i + 1 < len(rec_texts) and "Date of Birth" not in data:
                dob_text = rec_texts[i + 1].strip()
                data["Date of Birth"] = dob_text
                scores["Date of Birth"] = _score_at(rec_scores, i + 1)

        # Standalone date — could be DOB if name/father already found but DOB hasn't been
        elif _DATE_RE.fullmatch(t) and "Date of Birth" not in data:
            if "Name" in data:  # Only treat as DOB if we've already seen the name
                data["Date of Birth"] = t
                scores["Date of Birth"] = _score_at(rec_scores, i)

    return data, scores


def _canonicalize(raw: dict, raw_scores: dict) -> tuple:
    canonical, confidence = {}, {}
    s = raw_scores or {}

    def _set(canon_key, raw_key, value):
        canonical[canon_key] = value
        if s.get(raw_key) is not None:
            confidence[canon_key] = s[raw_key]

    canonical["document_type"] = "pan"

    if "PAN" in raw:
        _set("pan_number", "PAN", raw["PAN"].upper().strip())
    if "Name" in raw:
        _set("name", "Name", raw["Name"].strip())
    if "Father's Name" in raw:
        _set("fathers_name", "Father's Name", raw["Father's Name"].strip())
    if "Date of Birth" in raw:
        _set("date_of_birth", "Date of Birth", _to_iso_date(raw["Date of Birth"]))

    return canonical, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(rec_texts: list, rec_scores: list = None) -> tuple:
    """
    Extract and canonicalize PAN card fields from OCR output.

    Returns:
        (canonical: dict, confidence: dict)
    """
    rec_scores = rec_scores or []
    raw, scores = _extract_raw(rec_texts, rec_scores)
    return _canonicalize(raw, scores)


# ---------------------------------------------------------------------------
# PaddleOCRVL markdown extraction
# ---------------------------------------------------------------------------

def _search_pan(pattern: str, text: str, flags=re.IGNORECASE) -> str:
    """Return the first non-empty capture group from text, or empty string."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _extract_raw_from_markdown(md_text: str) -> dict:
    """
    Parse PAN card fields from PaddleOCRVL markdown output.

    PAN cards have a simple layout: the VL model renders the card as labelled
    lines or a small table, making straightforward regex reliable.
    """
    raw = {}

    # PAN number — 10-character alphanumeric in AAAAA9999A format
    pan_m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", md_text)
    if pan_m:
        raw["PAN"] = pan_m.group(1).upper()

    # Name of card holder
    name = _search_pan(
        r"(?:^|\n)\s*(?:Name|NAME)\s*[:\|]?\s*([A-Z][A-Za-z .'-]{1,79})(?:\n|$)",
        md_text,
        flags=re.MULTILINE,
    )
    # Fallback: line just below a line containing only "Name"
    if not name:
        name_m = re.search(
            r"(?:^|\n)(?:Name|NAME)\s*\n([A-Z][A-Za-z .'-]{1,79})(?:\n|$)",
            md_text,
            re.MULTILINE,
        )
        if name_m:
            name = name_m.group(1).strip()
    if name:
        raw["Name"] = name

    # Father's Name
    fathers = _search_pan(
        r"(?:Father(?:'s)?\s+Name|FATHER(?:'S)?\s+NAME)\s*[:\|]?\s*([A-Z][A-Za-z .'-]{1,79})(?:\n|$)",
        md_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not fathers:
        fathers_m = re.search(
            r"(?:Father(?:'s)?\s+Name|FATHER(?:'S)?\s+NAME)\s*\n([A-Z][A-Za-z .'-]{1,79})(?:\n|$)",
            md_text,
            re.MULTILINE | re.IGNORECASE,
        )
        if fathers_m:
            fathers = fathers_m.group(1).strip()
    if fathers:
        raw["Father's Name"] = fathers

    # Date of Birth
    dob = _search_pan(
        r"(?:Date\s+of\s+Birth|DOB|BIRTH\s+DATE)\s*[:\|]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        md_text,
    )
    if not dob:
        # Standalone date near bottom of card — grab first occurrence not already used
        dob_m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", md_text)
        if dob_m:
            dob = dob_m.group(1)
    if dob:
        raw["Date of Birth"] = dob

    return raw


def extract_from_markdown(md_text: str) -> tuple:
    """
    Extract and canonicalize PAN card fields from PaddleOCRVL markdown.

    Args:
        md_text: Full markdown string produced by PaddleOCRVL for the document.

    Returns:
        (canonical: dict, confidence: dict)
        confidence is empty because VL output does not carry per-field scores.
    """
    raw = _extract_raw_from_markdown(md_text)
    canonical, _ = _canonicalize(raw, {})
    return canonical, {}
