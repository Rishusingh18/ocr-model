"""
router.py
Auto-detects the document type from OCR text and routes to the correct extractor.

Usage (classic PaddleOCR path):
    from router import detect_document_type, route

    doc_type = detect_document_type(rec_texts)          # 'gst' | 'udyam' | 'pan' | 'unknown'
    canonical, confidence, doc_type = route(rec_texts, rec_scores)

Usage (PaddleOCRVL markdown path — preferred):
    from router import detect_document_type_from_markdown, route_markdown

    doc_type  = detect_document_type_from_markdown(md_text)
    canonical, confidence, doc_type = route_markdown(md_text)
"""
import re
from extractors import gst, udyam, pan

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# GST: 2-digit state code + 5 alpha + 4 digit + 1 alpha + 1 digit + Z + 1 alphanum
_GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")

# Udyam: UDYAM-XX-00-0000000
_UDYAM_RE = re.compile(r"UDYAM-[A-Z]{2}-\d{2}-\d{7}", re.IGNORECASE)

# PAN: AAAAA9999A
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


# ---------------------------------------------------------------------------
# Classic path (flat rec_texts list)
# ---------------------------------------------------------------------------

def detect_document_type(rec_texts: list) -> str:
    """
    Scan OCR tokens for document-type markers.

    Returns one of: 'gst' | 'udyam' | 'pan' | 'unknown'

    Detection priority:
        1. Udyam  — most distinctive (UDYAM- prefix)
        2. GST    — GSTIN pattern or "Form GST REG" header
        3. PAN    — PAN pattern or "Income Tax Department"
    """
    joined = " ".join(rec_texts)

    # --- Udyam (highest specificity) ---
    if _UDYAM_RE.search(joined):
        return "udyam"
    if any("Udyam Registration" in t for t in rec_texts):
        return "udyam"

    # --- GST ---
    if _GSTIN_RE.search(joined):
        return "gst"
    if any("Form GST REG" in t or "GST Registration" in t for t in rec_texts):
        return "gst"

    # --- PAN ---
    pan_hits = [t for t in rec_texts if _PAN_RE.fullmatch(t.strip())]
    if pan_hits:
        return "pan"
    if any("Income Tax Department" in t or "Permanent Account Number" in t for t in rec_texts):
        return "pan"

    return "unknown"


def route(rec_texts: list, rec_scores: list = None, doc_type: str = None) -> tuple:
    """
    Run the appropriate extractor for the given document (classic token path).

    Args:
        rec_texts:  OCR text tokens.
        rec_scores: Parallel confidence scores. Optional.
        doc_type:   Force a specific type ('gst'|'udyam'|'pan').
                    If None, auto-detects from rec_texts.

    Returns:
        (canonical: dict, confidence: dict, detected_type: str)
    """
    if doc_type is None:
        doc_type = detect_document_type(rec_texts)

    if doc_type == "gst":
        canonical, confidence = gst.extract(rec_texts, rec_scores)
    elif doc_type == "udyam":
        canonical, confidence = udyam.extract(rec_texts, rec_scores)
    elif doc_type == "pan":
        canonical, confidence = pan.extract(rec_texts, rec_scores)
    else:
        # Unknown — return the raw OCR texts so the caller can inspect
        canonical = {"document_type": "unknown", "raw_texts": rec_texts}
        confidence = {}

    return canonical, confidence, doc_type


# ---------------------------------------------------------------------------
# PaddleOCRVL markdown path
# ---------------------------------------------------------------------------

def detect_document_type_from_markdown(md_text: str) -> str:
    """
    Detect document type from the structured markdown produced by PaddleOCRVL.

    Detection priority (same as classic path):
        1. Udyam  — UDYAM- number or "Udyam Registration" heading
        2. GST    — GSTIN pattern or "Form GST REG" / "GST Registration"
        3. PAN    — PAN number or "Income Tax Department" / "Permanent Account Number"

    Returns one of: 'gst' | 'udyam' | 'pan' | 'unknown'
    """
    # --- Udyam ---
    if _UDYAM_RE.search(md_text):
        return "udyam"
    if re.search(r"Udyam\s+Registration", md_text, re.IGNORECASE):
        return "udyam"

    # --- GST ---
    if _GSTIN_RE.search(md_text):
        return "gst"
    if re.search(r"(?:Form\s+GST\s+REG|GST\s+Registration\s+Certificate)", md_text, re.IGNORECASE):
        return "gst"

    # --- PAN ---
    if _PAN_RE.search(md_text):
        return "pan"
    if re.search(r"(?:Income\s+Tax\s+Department|Permanent\s+Account\s+Number)", md_text, re.IGNORECASE):
        return "pan"

    return "unknown"


def route_markdown(md_text: str, doc_type: str = None) -> tuple:
    """
    Run the appropriate extractor for the given document (VL markdown path).

    Args:
        md_text:  Structured markdown produced by PaddleOCRVL.
        doc_type: Force a specific type ('gst'|'udyam'|'pan').
                  If None, auto-detects from the markdown body.

    Returns:
        (canonical: dict, confidence: dict, detected_type: str)
    """
    if doc_type is None:
        doc_type = detect_document_type_from_markdown(md_text)

    if doc_type == "gst":
        canonical, confidence = gst.extract_from_markdown(md_text)
    elif doc_type == "udyam":
        canonical, confidence = udyam.extract_from_markdown(md_text)
    elif doc_type == "pan":
        canonical, confidence = pan.extract_from_markdown(md_text)
    else:
        # Unknown — surface the raw markdown so the caller can inspect
        canonical = {"document_type": "unknown", "raw_markdown": md_text}
        confidence = {}

    return canonical, confidence, doc_type
