"""
extractors/udyam.py
Extractor for Udyam Registration Certificates.

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
    """Convert DD/MM/YYYY, DD-MM-YYYY, or YYYY-MM-DD to ISO 8601."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _parse_address(address_str: str) -> dict:
    """Parse a flat address string into structured sub-fields."""
    label_map = {
        r"Flat/Door/Block No\.?":   "building",
        r"Village/Town":            "locality",
        r"Block(?!\s*/|\s*No)":     "block",
        r"Road/Street/Lane":        "street",
        r"City":                    "city",
        r"State":                   "state",
        r"District":                "district",
        r"PIN":                     "pin_code",
    }
    result = {}
    all_labels = "|".join(label_map.keys())
    for pattern, key in label_map.items():
        m = re.search(
            rf"{pattern}\s*[:\-]?\s*(.+?)(?=,\s*(?:{all_labels})\s*[:\-]|$)",
            address_str,
            re.IGNORECASE,
        )
        if m:
            result[key] = m.group(1).strip().rstrip(",").strip()
    return result if result else {"raw": address_str}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

# Udyam number pattern: UDYAM-XX-00-0000000
_UDYAM_RE = re.compile(r"UDYAM-[A-Z]{2}-\d{2}-\d{7}", re.IGNORECASE)


def _extract_raw(rec_texts, rec_scores):
    data, scores = {}, {}

    # Known field labels — used to prevent greedy capture across field boundaries
    _STOP_LABELS = {
        "Bank Account Number", "Bank A/C No", "IFS Code", "IFSC",
        "NIC Code", "NIC 2 Digit Code", "Major Activity",
        "Social Category", "Date of Registration", "Date of Commencement",
        "Date of Incorporation", "Gender", "Type of Enterprise",
        "Type of Organisation", "Name of Enterprise", "Name of Owner",
        "Udyam Registration Number", "Udyam Registration No",
    }

    _SKIP_HEADERS = {
        "Classification Year", "Enterprise Type", "Classification Date",
        "Name of Unit", "Name of Unit(s)", "Flat/Door/Block", "Village/Town",
        "Road/Street/Lane", "City", "Pin", "State", "District",
        "Block", "Name of Premises/Building", "SNo"
    }

    def _assign(key, i_idx, text_token, match_labels, validator=None):
        if key in data:
            return
        
        # 1. Try colon split
        parts = text_token.split(":")
        inline = parts[1].strip() if len(parts) > 1 else ""
        
        # 2. Try space-separated inline (remainder of string after label)
        if not inline:
            t_low = text_token.lower()
            for label in match_labels:
                if label in t_low:
                    idx = t_low.find(label)
                    remainder = text_token[idx + len(label):].strip()
                    remainder = remainder.lstrip("*: -")
                    if remainder:
                        inline = remainder
                    break

        if inline:
            if validator:
                if validator(inline):
                    data[key] = inline
                    scores[key] = _score_at(rec_scores, i_idx)
                    return
            elif inline.replace(" ", "").replace(".", "").lower() != "sno" and not any(lbl.lower() in inline.lower() for lbl in _STOP_LABELS):
                data[key] = inline
                scores[key] = _score_at(rec_scores, i_idx)
                return
        
        # 3. If validator provided, bounded forward search (up to 10 tokens)
        if validator:
            for j in range(1, 11):
                if i_idx + j < len(rec_texts):
                    candidate = rec_texts[i_idx + j].strip()
                    c_low = candidate.lower()
                    if any(lbl.lower() in c_low for lbl in _SKIP_HEADERS) or candidate.replace(" ", "").replace(".", "").lower() == "sno":
                        continue
                    if any(lbl.lower() in c_low for lbl in _STOP_LABELS):
                        break
                    if validator(candidate):
                        data[key] = candidate
                        scores[key] = _score_at(rec_scores, i_idx + j)
                        return
        else:
            # Fallback for unvalidated fields: bounded forward search (up to 4 tokens)
            for j in range(1, 5):
                if i_idx + j < len(rec_texts):
                    candidate = rec_texts[i_idx + j].strip()
                    if not candidate:
                        continue
                    
                    c_low = candidate.lower()
                    if any(lbl.lower() in c_low for lbl in _SKIP_HEADERS) or candidate.replace(" ", "").replace(".", "").lower() == "sno":
                        continue # Skip interleaved labels/headers
                    if any(lbl.lower() in c_low for lbl in _STOP_LABELS):
                        break # Terminate if we hit another known field label
                    
                    if len(candidate.replace(" ", "")) > 1:
                        data[key] = candidate
                        scores[key] = _score_at(rec_scores, i_idx + j)
                        return

    for i, text in enumerate(rec_texts):
        t = text.strip()
        t_low = t.lower()

        # Udyam Registration Number — appears inline or on the next line
        if "udyam registration number" in t_low or "udyam registration no" in t_low:
            # Try same-line first
            m = _UDYAM_RE.search(t)
            if m:
                data["Udyam Number"] = m.group(0).upper()
                scores["Udyam Number"] = _score_at(rec_scores, i)
            elif i + 1 < len(rec_texts):
                next_t = rec_texts[i + 1].strip()
                # Only store if the next token matches the expected pattern
                m2 = _UDYAM_RE.search(next_t)
                if m2:
                    data["Udyam Number"] = m2.group(0).upper()
                    scores["Udyam Number"] = _score_at(rec_scores, i + 1)
                # else: leave unset — an unvalidated token must not become the upsert key

        # Also catch standalone Udyam number anywhere in text
        elif _UDYAM_RE.match(t) and "Udyam Number" not in data:
            data["Udyam Number"] = t.upper()
            scores["Udyam Number"] = _score_at(rec_scores, i)

        elif any(x in t_low for x in ["name of enterprise", "name of the enterprise", "name of organisation", "name of the organisation"]):
            _assign("Enterprise Name", i, t, ["name of enterprise", "name of the enterprise", "name of organisation", "name of the organisation"])

        elif "name of owner" in t_low or "name of the owner" in t_low:
            _assign("Owner Name", i, t, ["name of owner", "name of the owner"])

        elif "type of enterprise" in t_low or "type of organisation" in t_low:
            _assign(
                "Type of Enterprise", i, t, ["type of enterprise", "type of organisation"],
                validator=lambda x: x.lower() in ["micro", "small", "medium"]
            )

        elif "major activity" in t_low:
            _assign("Major Activity", i, t, ["major activity"])

        elif "nic code" in t_low or "nic 2 digit code" in t_low or "national industry classification" in t_low:
            _assign("NIC Code", i, t, ["nic code", "nic 2 digit code", "national industry classification"])

        elif "social category" in t_low:
            _assign(
                "Social Category", i, t, ["social category"],
                validator=lambda x: x.lower() in ["general", "sc", "st", "obc"]
            )

        elif "gender" in t_low:
            _assign(
                "Gender", i, t, ["gender"],
                validator=lambda x: x.lower() in ["male", "female", "transgender"]
            )

        elif "date of incorporation" in t_low or "date of commencement" in t_low:
            _assign(
                "Date of Commencement", i, t, ["date of incorporation", "date of commencement"],
                validator=lambda x: bool(re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", x))
            )

        elif "date of registration" in t_low or "date of udyam registration" in t_low:
            _assign(
                "Date of Registration", i, t, ["date of registration", "date of udyam registration"],
                validator=lambda x: bool(re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", x))
            )

        elif "bank account number" in t_low or "bank a/c no" in t_low:
            _assign("Bank Account Number", i, t, ["bank account number", "bank a/c no"])

        elif "ifs code" in t_low or "ifsc" in t_low:
            _assign("IFSC Code", i, t, ["ifs code", "ifsc"])

        elif "official address" in t_low or re.fullmatch(r"address", t, re.IGNORECASE):
            address_lines, addr_scores = [], []
            j = i + 1
            while j < len(rec_texts):
                line = rec_texts[j].strip()
                if any(label in line for label in _STOP_LABELS):
                    break
                if line and len(line) > 2:
                    address_lines.append(line)
                    s = _score_at(rec_scores, j)
                    if s is not None:
                        addr_scores.append(s)
                j += 1
            if address_lines:
                data["Address"] = ", ".join(address_lines)
                scores["Address"] = round(min(addr_scores), 4) if addr_scores else None

    return data, scores


def _canonicalize(raw: dict, raw_scores: dict) -> tuple:
    canonical, confidence = {}, {}
    s = raw_scores or {}

    def _set(canon_key, raw_key, value):
        canonical[canon_key] = value
        if s.get(raw_key) is not None:
            confidence[canon_key] = s[raw_key]

    canonical["document_type"] = "udyam"

    if "Udyam Number" in raw:
        _set("udyam_number", "Udyam Number", raw["Udyam Number"].upper().strip())
    if "Enterprise Name" in raw:
        _set("enterprise_name", "Enterprise Name", raw["Enterprise Name"].strip())
    if "Owner Name" in raw:
        _set("owner_name", "Owner Name", raw["Owner Name"].strip())
    if "Type of Enterprise" in raw:
        _set("type_of_enterprise", "Type of Enterprise", raw["Type of Enterprise"].strip())
    if "Major Activity" in raw:
        _set("major_activity", "Major Activity", raw["Major Activity"].strip())
    if "NIC Code" in raw:
        _set("nic_code", "NIC Code", raw["NIC Code"].strip())
    if "Social Category" in raw:
        _set("social_category", "Social Category", raw["Social Category"].strip())
    if "Gender" in raw:
        _set("gender", "Gender", raw["Gender"].strip())
    if "Date of Registration" in raw:
        _set("date_of_registration", "Date of Registration",
             _to_iso_date(raw["Date of Registration"]))
    if "Date of Commencement" in raw:
        _set("date_of_commencement", "Date of Commencement",
             _to_iso_date(raw["Date of Commencement"]))
    if "Address" in raw:
        canonical["address"] = _parse_address(raw["Address"])
        if s.get("Address") is not None:
            confidence["address"] = s["Address"]
    if "Bank Account Number" in raw:
        _set("bank_account_number", "Bank Account Number", raw["Bank Account Number"].strip())
    if "IFSC Code" in raw:
        _set("ifsc_code", "IFSC Code", raw["IFSC Code"].strip())

    return canonical, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(rec_texts: list, rec_scores: list = None) -> tuple:
    """
    Extract and canonicalize Udyam Registration Certificate fields.

    Returns:
        (canonical: dict, confidence: dict)
    """
    rec_scores = rec_scores or []
    raw, scores = _extract_raw(rec_texts, rec_scores)
    return _canonicalize(raw, scores)


# ---------------------------------------------------------------------------
# PaddleOCRVL markdown extraction
# ---------------------------------------------------------------------------

def _search_md(pattern: str, text: str, flags=re.IGNORECASE) -> str:
    """Return the first non-empty capture group, or empty string."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _extract_raw_from_markdown(md_text: str) -> dict:
    """
    Parse Udyam fields from PaddleOCRVL markdown output.

    VL markdown keeps label and value on adjacent lines / in the same table cell,
    so regex is simpler and more reliable than the bounded-search used in
    _extract_raw() for flat rec_texts.
    """
    raw = {}

    # Udyam Registration Number
    udyam = _search_md(
        r"(UDYAM-[A-Z]{2}-\d{2}-\d{7})",
        md_text,
        flags=re.IGNORECASE,
    )
    if udyam:
        raw["Udyam Number"] = udyam.upper()

    # Enterprise / Organisation Name
    enterprise = _search_md(
        r"(?:Name\s+of\s+(?:the\s+)?(?:Enterprise|Organisation|Unit))\s*[:\|]?\s*([^\n|]{3,120})",
        md_text,
    )
    if enterprise:
        raw["Enterprise Name"] = enterprise

    # Owner Name
    owner = _search_md(
        r"(?:Name\s+of\s+(?:the\s+)?Owner)\s*[:\|]?\s*([^\n|]{3,80})",
        md_text,
    )
    if owner:
        raw["Owner Name"] = owner

    # Type of Enterprise / Organisation
    ent_type = _search_md(
        r"Type\s+of\s+(?:Enterprise|Organisation)\s*[:\|]?\s*(Micro|Small|Medium)",
        md_text,
    )
    if ent_type:
        raw["Type of Enterprise"] = ent_type

    # Major Activity
    activity = _search_md(
        r"Major\s+Activity\s*[:\|]?\s*([^\n|]{3,60})",
        md_text,
    )
    if activity:
        raw["Major Activity"] = activity

    # NIC Code
    nic = _search_md(
        r"(?:NIC\s+(?:2\s+Digit\s+)?Code|National\s+Industry\s+Classification)\s*[:\|]?\s*([^\n|]{1,30})",
        md_text,
    )
    if nic:
        raw["NIC Code"] = nic

    # Social Category
    category = _search_md(
        r"Social\s+Category\s*[:\|]?\s*(General|SC|ST|OBC)",
        md_text,
    )
    if category:
        raw["Social Category"] = category

    # Gender
    gender = _search_md(
        r"Gender\s*[:\|]?\s*(Male|Female|Transgender)",
        md_text,
    )
    if gender:
        raw["Gender"] = gender

    # Date of Registration
    date_reg = _search_md(
        r"Date\s+of\s+(?:Udyam\s+)?Registration\s*[:\|]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        md_text,
    )
    if date_reg:
        raw["Date of Registration"] = date_reg

    # Date of Commencement / Incorporation
    date_comm = _search_md(
        r"Date\s+of\s+(?:Incorporation|Commencement)\s*[:\|]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        md_text,
    )
    if date_comm:
        raw["Date of Commencement"] = date_comm

    # Bank Account Number
    bank_acc = _search_md(
        r"(?:Bank\s+(?:Account|A/C)\s+(?:Number|No\.?))\s*[:\|]?\s*([0-9]{9,18})",
        md_text,
    )
    if bank_acc:
        raw["Bank Account Number"] = bank_acc

    # IFSC Code
    ifsc = _search_md(
        r"(?:IFS\s*Code|IFSC)\s*[:\|]?\s*([A-Z]{4}0[A-Z0-9]{6})",
        md_text,
        flags=re.IGNORECASE,
    )
    if ifsc:
        raw["IFSC Code"] = ifsc.upper()

    # Address
    addr_m = re.search(
        r"(?:Official\s+)?Address\s*[:\|]?\s*([\s\S]{10,500}?)(?=\n\s*\n|Bank|NIC\s+Code|Social|$)",
        md_text,
        re.IGNORECASE,
    )
    if addr_m:
        raw["Address"] = " ".join(addr_m.group(1).split())

    return raw


def extract_from_markdown(md_text: str) -> tuple:
    """
    Extract and canonicalize Udyam certificate fields from PaddleOCRVL markdown.

    Args:
        md_text: Full markdown string produced by PaddleOCRVL for the document.

    Returns:
        (canonical: dict, confidence: dict)
        confidence is empty because VL output does not carry per-field scores.
    """
    raw = _extract_raw_from_markdown(md_text)
    canonical, _ = _canonicalize(raw, {})
    return canonical, {}
