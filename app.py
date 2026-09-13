"""
app.py
FastAPI server for the Parakh document OCR & extraction service.

Endpoints:
    POST /process-document   Unified endpoint — auto-detects document type.
    POST /process-gst        Backward-compatible GST-only endpoint.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import os
import logging
import functools
from uuid import uuid4

from pipeline import process_document, ocr_lock, get_vl

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parakh Document OCR API",
    description=(
        "Automated OCR & structured extraction for GST Certificates, "
        "Udyam Certificates, and PAN Cards using PaddleOCRVL. "
        "Extracted data is persisted to MongoDB in canonical form."
    ),
    version="3.0.0",
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Maximum allowed upload size (10 MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHUNK_SIZE = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# Shared upload helper
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> str:
    """
    Save an uploaded file to a unique temp path with a size cap.
    Returns the temp file path. Raises HTTPException on oversized uploads.
    Any I/O exception triggers cleanup of the partial temp file.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{ext}")

    total_written = 0
    try:
        with open(temp_path, "wb") as buf:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_BYTES:
                    buf.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
                    )
                buf.write(chunk)
    except HTTPException:
        raise  # already cleaned up above
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return temp_path


async def _run_ocr(temp_path: str, doc_type: str = None) -> dict:
    """
    Resolve the VL singleton (async), then dispatch the CPU-bound inference
    to the thread pool under the OCR lock.
    """
    vl = await get_vl()
    fn = functools.partial(process_document, temp_path, vl, doc_type)
    async with ocr_lock:
        result = await run_in_threadpool(fn)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/process-document",
    summary="Process any supported document (GST, Udyam, PAN)",
    response_description="Canonical extracted fields + confidence scores + DB result",
)
async def process_document_endpoint(
    file: UploadFile = File(...),
):
    """
    Upload a PDF or image document. The API will:
    1. Run PaddleOCRVL (full pipeline: layout detection + VLM).
    2. Auto-detect the document type (GST, Udyam, or PAN).
    3. Extract fields into a canonical schema.
    4. Persist the record to MongoDB.
    5. Return the canonical JSON.

    Note: The first request triggers VLM model loading (~30–90 s depending on
    hardware). Subsequent requests reuse the cached model.
    """
    temp_path = await _save_upload(file)
    try:
        result = await _run_ocr(temp_path)
        if result.get("status") == "error":
            msg = result.get("message", "")
            if "no text" in msg.lower() or "no output" in msg.lower():
                raise HTTPException(status_code=422, detail=msg)
            logger.error("process_document error: %s", msg)
            raise HTTPException(status_code=500, detail="Internal processing error.")
        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post(
    "/process-gst",
    summary="[Legacy] Process a GST Certificate",
    response_description="Canonical GST fields + confidence scores + DB result",
)
async def process_gst(file: UploadFile = File(...)):
    """
    Backward-compatible endpoint that forces doc_type='gst'.
    Prefer /process-document for new integrations.
    """
    temp_path = await _save_upload(file)
    try:
        result = await _run_ocr(temp_path, doc_type="gst")
        if result.get("status") == "error":
            msg = result.get("message", "")
            if "no text" in msg.lower() or "no output" in msg.lower():
                raise HTTPException(status_code=422, detail=msg)
            logger.error("process_document error: %s", msg)
            raise HTTPException(status_code=500, detail="Internal processing error.")
        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
