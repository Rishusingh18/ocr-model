"""
extractors/gst.py
Extractor for GST Registration Certificates (Form GST REG-06).

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
    """Return rounded confidence at index, or None if unavailable."""
    try:
        return round(float(scores[idx]), 4)
    except (IndexError, TypeError, ValueError):
        return None


def _to_iso_date(date_str: str) -> str:
    """Convert DD/MM/YYYY or DD-MM-YYYY to ISO 8601 YYYY-MM-DD."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _parse_address(address_str: str) -> dict:
    """
    Parse the flat OCR address string into a structured sub-object.
    e.g. "Building No./Flat No.: X, Road/Street: Y, City/Town/Village: Z ..."
    """
    label_map = {
        r"Floor No\.?":               "floor",
        r"Building No\.?/Flat No\.?": "building",
        r"Name Of Premises/Building": "premises",
        r"Road/Street":               "street",
        r"Nearby Landmark":           "landmark",
        r"Locality/Sub Locality":     "locality",
        r"City/Town/Village":         "city",
        r"District":                  "district",
        r"State":                     "state",
        r"PIN Code":                  "pin_code",
    }
    result = {}
    all_labels = "|".join(label_map.keys())
    for pattern, key in label_map.items():
        m = re.search(
            rf"{pattern}\s*:\s*(.+?)(?=,\s*(?:{all_labels})\s*:|$)",
            address_str,
            re.IGNORECASE,
        )
        if m:
            result[key] = m.group(1).strip().rstrip(",").strip()
    return result if result else {"raw": address_str}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _extract_raw(rec_texts, rec_scores):
    """
    Scan OCR tokens and map them to raw GST field names.
    First-occurrence-wins: once a key is set it is never overwritten.
    Returns (data: dict, scores: dict).
    """
    data, scores = {}, {}

    for i, text in enumerate(rec_texts):
        text = text.strip()

        if (
            "Registration Number" in text
            or "GSTIN" in text
            or "Goods and Services Tax Identification Number" in text
        ) and "GSTIN" not in data:
            parts = text.split(":")
            inline_value = parts[1].strip() if len(parts) > 1 else ""
            if inline_value:
                data["GSTIN"] = inline_value
                scores["GSTIN"] = _score_at(rec_scores, i)
            elif i + 1 < len(rec_texts):
                data["GSTIN"] = rec_texts[i + 1].strip()
                scores["GSTIN"] = _score_at(rec_scores, i + 1)

        elif "Legal Name" in text and "Legal Name" not in data:
            if i + 1 < len(rec_texts):
                data["Legal Name"] = rec_texts[i + 1].strip()
                scores["Legal Name"] = _score_at(rec_scores, i + 1)

        elif "Trade Name" in text and "Trade Name" not in data:
            if i + 1 < len(rec_texts):
                data["Trade Name"] = rec_texts[i + 1].strip()
                scores["Trade Name"] = _score_at(rec_scores, i + 1)

        elif "Constitution of Business" in text and "Constitution of Business" not in data:
            if i + 1 < len(rec_texts):
                data["Constitution of Business"] = rec_texts[i + 1].strip()
                scores["Constitution of Business"] = _score_at(rec_scores, i + 1)

        elif "Address of Principal Place" in text and "Address" not in data:
            address_lines, addr_scores = [], []
            j = i + 1
            while j < len(rec_texts):
                line = rec_texts[j].strip()
                if line.startswith("6.") or "Date of Liability" in line:
                    break
                if line and not line.startswith("Business") and line != "Address":
                    address_lines.append(line)
                    s = _score_at(rec_scores, j)
                    if s is not None:
                        addr_scores.append(s)
                j += 1
            data["Address"] = ", ".join(address_lines)
            scores["Address"] = round(min(addr_scores), 4) if addr_scores else None

        elif "Type of Registration" in text and "Type of Registration" not in data:
            if i + 1 < len(rec_texts):
                data["Type of Registration"] = rec_texts[i + 1].strip()
                scores["Type of Registration"] = _score_at(rec_scores, i + 1)

        elif "Date of issue of Certificate" in text and "Date of Issue" not in data:
            if i + 1 < len(rec_texts):
                data["Date of Issue"] = rec_texts[i + 1].strip()
                scores["Date of Issue"] = _score_at(rec_scores, i + 1)

    return data, scores


def _canonicalize(raw: dict, raw_scores: dict) -> tuple:
    """Convert raw field dict to canonical snake_case schema + confidence map."""
    canonical, confidence = {}, {}
    s = raw_scores or {}

    def _set(canon_key, raw_key, value):
        canonical[canon_key] = value
        if s.get(raw_key) is not None:
            confidence[canon_key] = s[raw_key]

    canonical["document_type"] = "gst"

    if "GSTIN" in raw:
        _set("gstin", "GSTIN", raw["GSTIN"].upper().strip())
    if "Legal Name" in raw:
        _set("legal_name", "Legal Name", raw["Legal Name"].strip())
    if "Trade Name" in raw:
        _set("trade_name", "Trade Name", raw["Trade Name"].strip())
    if "Constitution of Business" in raw:
        _set("constitution_of_business", "Constitution of Business",
             raw["Constitution of Business"].strip())
    if "Address" in raw:
        canonical["address"] = _parse_address(raw["Address"])
        if s.get("Address") is not None:
            confidence["address"] = s["Address"]
    if "Type of Registration" in raw:
        _set("type_of_registration", "Type of Registration",
             raw["Type of Registration"].strip())
    if "Date of Issue" in raw:
        _set("date_of_issue", "Date of Issue", _to_iso_date(raw["Date of Issue"]))

    return canonical, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(rec_texts: list, rec_scores: list = None) -> tuple:
    """
    Extract and canonicalize GST certificate fields from OCR output.

    Args:
        rec_texts:  List of recognized text strings from PaddleOCR.
        rec_scores: Parallel list of confidence floats (0-1). Optional.

    Returns:
        (canonical: dict, confidence: dict)
        canonical  — rule-engine-ready field dict with document_type="gst"
        confidence — per-field OCR confidence scores (subset of canonical keys)
    """
    rec_scores = rec_scores or []
    raw, scores = _extract_raw(rec_texts, rec_scores)
    return _canonicalize(raw, scores)


# ---------------------------------------------------------------------------
# PaddleOCRVL markdown extraction
# ---------------------------------------------------------------------------

def _search(pattern: str, text: str, flags=re.IGNORECASE) -> str:
    """Return the first non-empty capture group, or empty string."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _extract_raw_from_markdown(md_text: str) -> dict:
    """
    Parse GST fields from PaddleOCRVL markdown output.

    Markdown from VL preserves structural relationships (labels near values,
    table cells) far better than flat rec_texts, so simple regex is sufficient.
    """
    raw = {}

    # GSTIN: 15-char alphanum matching GST format
    gstin = _search(
        r"(?:GSTIN|GST(?:IN)?\s*(?:No\.?|Number)?|Registration\s+Number)\s*[:\|]?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])",
        md_text,
    )
    if not gstin:
        # Standalone GSTIN anywhere in text
        m = re.search(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b", md_text)
        if m:
            gstin = m.group(1)
    if gstin:
        raw["GSTIN"] = gstin

    # Legal Name
    legal = _search(
        r"Legal\s+Name\s+(?:of\s+(?:Business|Proprietor)\s+)?[:\|]?\s*([^\n|]{3,80})",
        md_text,
    )
    if legal:
        raw["Legal Name"] = legal

    # Trade Name
    trade = _search(
        r"Trade\s+Name\s*[:\|]?\s*([^\n|]{3,80})",
        md_text,
    )
    if trade:
        raw["Trade Name"] = trade

    # Constitution of Business
    constitution = _search(
        r"Constitution\s+of\s+Business\s*[:\|]?\s*([^\n|]{3,60})",
        md_text,
    )
    if constitution:
        raw["Constitution of Business"] = constitution

    # Type of Registration
    reg_type = _search(
        r"Type\s+of\s+Registration\s*[:\|]?\s*([^\n|]{3,60})",
        md_text,
    )
    if reg_type:
        raw["Type of Registration"] = reg_type

    # Date of Issue
    date_of_issue = _search(
        r"Date\s+of\s+(?:Issue|issue\s+of\s+Certificate)\s*[:\|]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        md_text,
    )
    if date_of_issue:
        raw["Date of Issue"] = date_of_issue

    # Address — everything after "Principal Place" until next section
    addr_m = re.search(
        r"(?:Address\s+of\s+)?Principal\s+Place\s+of\s+Business\s*[:\|]?\s*([\s\S]{10,400}?)(?=\n\s*\n|\d+\.|Type\s+of\s+Reg|Date\s+of|$)",
        md_text,
        re.IGNORECASE,
    )
    if addr_m:
        raw["Address"] = " ".join(addr_m.group(1).split())

    return raw


def extract_from_markdown(md_text: str) -> tuple:
    """
    Extract and canonicalize GST certificate fields from PaddleOCRVL markdown.

    Args:
        md_text: Full markdown string produced by PaddleOCRVL for the document.

    Returns:
        (canonical: dict, confidence: dict)
        confidence is empty because VL output does not carry per-field scores.
    """
    raw = _extract_raw_from_markdown(md_text)
    canonical, _ = _canonicalize(raw, {})
    return canonical, {}
