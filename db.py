"""
db.py
MongoDB persistence layer for extracted document data.

Each document type maps to its own collection and upserts by its primary key,
so re-uploading the same document updates the record rather than duplicating it.

    Collection          Primary key (upsert field)
    ───────────────     ──────────────────────────
    gst_certificates    gstin
    udyam_certificates  udyam_number
    pan_cards           pan_number

Usage:
    from db import save_document

    result = save_document(canonical_dict)
    # result: {"saved": True, "collection": "gst_certificates", "upserted": True}
"""
import logging
from datetime import datetime, timezone

from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "parakh_db"

# document_type -> (collection name, primary key field)
_COLLECTION_MAP = {
    "gst":   ("gst_certificates",   "gstin"),
    "udyam": ("udyam_certificates", "udyam_number"),
    "pan":   ("pan_cards",          "pan_number"),
}

# ---------------------------------------------------------------------------
# Connection (lazy singleton)
# ---------------------------------------------------------------------------

_client = None


def _get_db():
    """Return a MongoDB database handle, creating the client on first call."""
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[DB_NAME]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_document(canonical: dict) -> dict:
    """
    Upsert a canonicalized document into the appropriate MongoDB collection.

    Args:
        canonical: Dict produced by any extractor (must contain 'document_type').

    Returns:
        {
          "saved":       True | False,
          "collection":  "<collection name>",
          "primary_key": "<key field>",
          "upserted":    True  (new record) | False (updated existing),
          "error":       "<message>"  # only present on failure
        }
    """
    doc_type = canonical.get("document_type", "unknown")

    if doc_type not in _COLLECTION_MAP:
        return {
            "saved": False,
            "error": f"No collection mapping for document_type='{doc_type}'",
        }

    collection_name, pk_field = _COLLECTION_MAP[doc_type]
    pk_value = canonical.get(pk_field)

    if not pk_value:
        return {
            "saved": False,
            "collection": collection_name,
            "primary_key": pk_field,
            "error": f"Primary key '{pk_field}' is missing or empty — cannot upsert.",
        }

    try:
        db = _get_db()
        col = db[collection_name]

        # Stamp the record with last-updated time
        doc_to_save = {**canonical, "_updated_at": datetime.now(timezone.utc).isoformat()}

        result = col.update_one(
            {pk_field: pk_value},
            {"$set": doc_to_save},
            upsert=True,
        )

        was_upserted = result.upserted_id is not None
        logger.info(
            "Saved to %s [key=%s] upserted=%s",
            collection_name, pk_field, was_upserted,
        )

        return {
            "saved":       True,
            "collection":  collection_name,
            "primary_key": pk_field,
            "upserted":    was_upserted,
        }

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.warning("MongoDB unavailable: %s", e)
        return {
            "saved":      False,
            "collection": collection_name,
            "primary_key": pk_field,
            "error":      "MongoDB unavailable — is it running on localhost:27017?",
        }
    except Exception as e:
        logger.exception("Unexpected DB error")
        return {
            "saved":      False,
            "collection": collection_name,
            "primary_key": pk_field,
            "error":      str(e),
        }
