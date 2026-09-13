"""
pipeline.py
Central OCR orchestrator using PaddleOCRVL — the full Vision-Language pipeline
with layout detection, document unwarping, and seal recognition enabled.

Workflow:
    1. Pre-process the uploaded file (PDF → 300 DPI images via PyMuPDF + sharpening).
    2. Run PaddleOCRVL on each page — layout detection segments the page into
       regions (text, table, figure, title …), the VLM then reads each region
       and produces structured Markdown output.
    3. Concatenate per-page markdown into a single document-level string.
    4. Auto-detect document type via router.py (detect_document_type_from_markdown).
    5. Route to the correct extractor's extract_from_markdown() for canonical output.
    6. Optionally persist to MongoDB via db.py.

Model loading strategy — lazy init:
    PaddleOCRVL is NOT loaded at import time. It is created on the first OCR
    request and cached for the lifetime of the process. This keeps startup
    instant and avoids RAM conflicts with any other processes that may be
    running. The ocr_lock serialises concurrent requests as before.
"""
import json
import tempfile
import os
import logging
import asyncio

import cv2
import numpy as np
import fitz  # PyMuPDF

# Skip slow connectivity check at startup so the server boots fast; models
# that are already cached load from disk, new ones are downloaded on demand.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCRVL
from router import route_markdown
from db import save_document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded PaddleOCRVL singleton
# ---------------------------------------------------------------------------
# The model is intentionally NOT loaded at module import time.
# _get_vl() creates it on the first OCR call and caches it in _vl_instance.
# This prevents RAM conflicts during server startup when other heavyweight
# processes may be running.

_vl_instance: PaddleOCRVL | None = None

# Asyncio lock: VLM is not thread-safe for concurrent inference.
# app.py acquires this lock before dispatching to run_in_threadpool.
ocr_lock = asyncio.Lock()


def _build_vl() -> PaddleOCRVL:
    """
    Construct and return a fully-featured PaddleOCRVL instance.

    All sub-pipelines enabled for maximum structure recognition on Indian
    government documents (GST REG-06, Udyam certificates, PAN cards):

    use_layout_detection   — PP-DocLayoutV3 segments the page into semantic
                             regions (text, table, figure, title) before the
                             VLM reads each region. This is the biggest quality
                             improvement for multi-column / tabular documents.
    use_doc_orientation_classify — auto-rotates upside-down / sideways scans.
    use_doc_unwarping      — UVDoc corrects perspective distortion in photos.
    use_seal_recognition   — reads circular stamps on some Udyam certificates.
    use_ocr_for_image_block — runs OCR over embedded image regions.
    """
    print("Loading PaddleOCRVL (full pipeline) into memory…")
    vl = PaddleOCRVL(
        use_layout_detection=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_seal_recognition=True,
        use_chart_recognition=False,    # no charts in these document types
        use_ocr_for_image_block=True,
    )
    print("PaddleOCRVL ready.")
    return vl


async def get_vl() -> PaddleOCRVL:
    """Return the cached VL instance, creating it on the first call."""
    global _vl_instance
    if _vl_instance is None:
        loop = asyncio.get_event_loop()
        _vl_instance = await loop.run_in_executor(None, _build_vl)
    return _vl_instance


# ---------------------------------------------------------------------------
# Markdown extraction from VL results
# ---------------------------------------------------------------------------

def _get_markdown(vl_results: list) -> tuple[str, int]:
    """
    Collect and concatenate the structured markdown from PaddleOCRVL results.

    Each element in vl_results corresponds to one page image. We try the
    idiomatic .markdown attribute first, then .to_markdown(), then fall back
    to the JSON representation for older releases.

    Returns:
        (markdown_text: str, pages_processed: int)
    """
    pages_md = []
    for page in vl_results:
        if page is None:
            continue

        md = None
        if hasattr(page, "markdown"):
            md = page.markdown
        elif hasattr(page, "to_markdown"):
            try:
                md = page.to_markdown()
            except Exception:
                pass

        if not md:
            # Fallback: dump to JSON and stitch text fields together
            try:
                path = tempfile.mktemp(suffix=".json")
                page.save_to_json(path)
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                os.remove(path)
                texts = raw.get("rec_texts", raw.get("texts", raw.get("rec_text", [])))
                md = "\n".join(texts) if isinstance(texts, list) else str(raw)
            except Exception:
                continue

        if md and md.strip():
            pages_md.append(md.strip())

    return "\n\n".join(pages_md), len(pages_md)


# ---------------------------------------------------------------------------
# Image enhancement (PDF rendering + sharpening)
# ---------------------------------------------------------------------------

def _enhance_for_ocr(file_path: str, tmp_dir: str) -> list:
    """
    Render PDFs to 300 DPI PNGs and sharpen all images for better OCR.
    Returns a list of absolute paths to the prepared image files.
    """
    paths = []
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            zoom = 300 / 72          # 300 DPI
            MAX_PAGES   = 5
            MAX_PIXELS  = 16_000_000

            for i in range(min(len(doc), MAX_PAGES)):
                page = doc.load_page(i)
                rect = page.rect
                w, h = rect.width * zoom, rect.height * zoom
                mat = (
                    fitz.Matrix((MAX_PIXELS / (rect.width * rect.height)) ** 0.5,
                                (MAX_PIXELS / (rect.width * rect.height)) ** 0.5)
                    if w * h > MAX_PIXELS
                    else fitz.Matrix(zoom, zoom)
                )
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out = os.path.join(tmp_dir, f"page_{i}.png")
                pix.save(out)
                paths.append(out)
        else:
            paths.append(file_path)

        # Upscale 2× and sharpen every image
        final_paths = []
        for idx, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                final_paths.append(p)
                continue

            MAX_PIXELS = 16_000_000
            h, w = img.shape[:2]
            th, tw = h * 2, w * 2
            if th * tw > MAX_PIXELS:
                scale = (MAX_PIXELS / (h * w)) ** 0.5
                th, tw = (int(h * scale), int(w * scale)) if scale > 1 else (h, w)

            up = cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            sharpened = cv2.filter2D(
                gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            )
            out_p = os.path.join(tmp_dir, f"enhanced_{idx}.png")
            cv2.imwrite(out_p, sharpened)
            final_paths.append(out_p)

        return final_paths
    except Exception as e:
        logger.warning("Enhancement failed: %s. Falling back to original.", e)
        return [file_path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(file_path: str, vl: PaddleOCRVL, doc_type: str = None) -> dict:
    """
    Run PaddleOCRVL on *file_path* and return a canonical extraction result.

    Args:
        file_path: Absolute path to a PDF or image file.
        vl:        The PaddleOCRVL instance (resolved in the async context
                   before this function is dispatched to the thread pool).
        doc_type:  Force document type ('gst'|'udyam'|'pan').
                   If None, the type is auto-detected from the VL markdown.

    Returns a dict with keys:
        status            'success' | 'error'
        document_type     detected/forced type string
        pages_processed   number of pages processed
        extracted_data    canonical field dict
        confidence_scores per-field scores (empty for VL path)
        db_result         MongoDB persistence result
        message           present only on error
    """
    try:
        vl_results = []
        with tempfile.TemporaryDirectory() as tmp:
            enhanced_paths = _enhance_for_ocr(file_path, tmp)
            for ep in enhanced_paths:
                results = vl.predict(ep)
                if results:
                    vl_results.extend(results)

        if not vl_results:
            return {"status": "error", "message": "OCR produced no output for this file."}

        md_text, pages_processed = _get_markdown(vl_results)

        if not md_text.strip():
            return {"status": "error", "message": "OCR produced no text for this file."}

        canonical, confidence, detected_type = route_markdown(md_text, doc_type)
        db_result = save_document(canonical)

        return {
            "status":            "success",
            "document_type":     detected_type,
            "pages_processed":   pages_processed,
            "extracted_data":    canonical,
            "confidence_scores": confidence,
            "db_result":         db_result,
        }

    except Exception as e:
        logger.exception("process_document failed for %s", file_path)
        return {"status": "error", "message": str(e)}
