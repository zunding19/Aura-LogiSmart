import json
import math
from datetime import datetime, timezone
from uuid import uuid4

from src.preprocessing import ShipmentValidationError


def _nonblank_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShipmentValidationError(
            f"{field} must be a nonblank text value."
        )
    return value.strip()


def record_custody_handshake(
    shipment_id,
    releasing_handler,
    receiving_handler,
    buffer_hours,
    transfer_point,
    confirmed_by_both,
    records=None,
) -> dict:
    shipment_id = _nonblank_text(
        shipment_id, "shipment_id"
    )
    releasing_handler = _nonblank_text(
        releasing_handler, "releasing_handler"
    )
    receiving_handler = _nonblank_text(
        receiving_handler, "receiving_handler"
    )
    transfer_point = _nonblank_text(
        transfer_point, "transfer_point"
    )

    if (
        releasing_handler.casefold()
        == receiving_handler.casefold()
    ):
        raise ShipmentValidationError(
            "Releasing and receiving handlers must be distinct."
        )

    if confirmed_by_both is not True:
        raise ShipmentValidationError(
            "Both handlers' confirmation is required "
            "to record the handshake."
        )

    if isinstance(buffer_hours, bool):
        raise ShipmentValidationError(
            "buffer_hours must be a finite nonnegative number."
        )

    try:
        buffer = float(buffer_hours)
    except (TypeError, ValueError):
        raise ShipmentValidationError(
            "buffer_hours must be a finite nonnegative number."
        ) from None

    if not math.isfinite(buffer) or buffer < 0:
        raise ShipmentValidationError(
            "buffer_hours must be a finite nonnegative number."
        )

    if records is not None and not isinstance(records, list):
        raise ShipmentValidationError(
            "records must be a list when supplied."
        )

    record = {
        "record_id": f"PROTOTYPE-{uuid4().hex}",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "shipment_id": shipment_id,
        "releasing_handler": releasing_handler,
        "receiving_handler": receiving_handler,
        "buffer_hours": buffer,
        "transfer_point": transfer_point,
        "confirmed_by_both": True,
        "record_type": (
            "User-attested local prototype custody record"
        ),
        "note": (
            "No independent identity, package-condition, "
            "compliance or tamperproof verification."
        ),
    }

    if records is not None:
        records.append(dict(record))

    return record


def export_custody_audit(records) -> str:
    if (
        not isinstance(records, list)
        or any(
            not isinstance(item, dict)
            for item in records
        )
    ):
        raise ShipmentValidationError(
            "Supply a list of custody-record dictionaries."
        )

    try:
        return json.dumps(
            {
                "record_type": "Local prototype audit export",
                "warning": (
                    "User attestations only; "
                    "no identity verification "
                    "or tamperproof guarantee."
                ),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ShipmentValidationError(
            "Custody records must contain finite, "
            "JSON-serializable values."
        ) from None