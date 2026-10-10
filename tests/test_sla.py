"""Smart SLA rule and the custody handshake that releases it."""
import json

import pandas as pd
import pytest

from src.intake import assess_order
from src.model import load_model
from src.preprocessing import ShipmentValidationError
from src.protocols import export_custody_audit, record_custody_handshake
from src.recommendations import check_sla_status


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.mark.parametrize(
    "risk,buffer,triggered",
    [
        (.95, 1.0, True),
        (.81, 2.9, True),
        (.80, 1.0, False),   # must be above 80%
        (.95, 3.0, False),   # must be under 3.0 hours
        (.95, None, False),  # no buffer known
        (.50, 0.5, False),
    ],
)
def test_sla_rule(risk, buffer, triggered):
    status = check_sla_status(risk, buffer)

    assert status["trigger_condition"] is triggered
    assert status["simulated_escrow_locked"] is triggered
    assert (status["breach_notice"] is not None) is triggered


def test_handshake_with_corrective_action_releases_the_escrow():
    log = []
    locked = check_sla_status(
        .95, 1.0, shipment_id="SC-1", custody_records=log
    )
    assert locked["simulated_escrow_locked"]
    assert "SC-1" == locked["breach_notice"]["shipment_id"]

    # A plain handover is not corrective proof.
    record_custody_handshake("SC-1", "Hub A", "Hub B", 1.0, "Dubai", True, log)
    assert check_sla_status(
        .95, 1.0, shipment_id="SC-1", custody_records=log
    )["simulated_escrow_locked"]

    # Re-icing logged for another shipment does not count either.
    record_custody_handshake(
        "SC-2", "Hub A", "Hub B", 48.0, "Dubai", True, log,
        corrective_action="refrigerant_refresh",
    )
    # Nor does a corrective action that leaves the buffer under 3 hours.
    record_custody_handshake(
        "SC-1", "Hub A", "Hub B", 2.0, "Dubai", True, log,
        corrective_action="refrigerant_refresh",
    )
    assert check_sla_status(
        .95, 1.0, shipment_id="SC-1", custody_records=log
    )["simulated_escrow_locked"]

    record_custody_handshake(
        "SC-1", "Hub A", "Hub B", 49.0, "Dubai", True, log,
        corrective_action="refrigerant_refresh",
    )
    released = check_sla_status(
        .95, 1.0, shipment_id="SC-1", custody_records=log
    )

    assert released["trigger_condition"]
    assert not released["simulated_escrow_locked"]
    assert released["corrective_proof"]["corrective_action"] == "refrigerant_refresh"
    assert "released" in released["status"].lower()
    assert len(json.loads(export_custody_audit(log))["records"]) == 4


def test_handshake_rejects_unknown_corrective_action_and_one_sided_sign_off():
    with pytest.raises(ShipmentValidationError):
        record_custody_handshake(
            "SC-1", "A", "B", 5, "Dubai", True, corrective_action="prayer"
        )
    with pytest.raises(ShipmentValidationError):
        record_custody_handshake("SC-1", "A", "B", 5, "Dubai", False)


def test_assessment_carries_the_sla_and_reacts_to_the_handshake(model):
    data = pd.read_csv("data/processed/aura_shipments.csv")
    order = data[data.Shipment_ID.eq("SC-12132")].iloc[0]

    log = []
    before = assess_order(order, model, custody_records=log)
    assert before["sla"]["simulated_escrow_locked"]

    record_custody_handshake(
        "SC-12132", "Dubai Air Hub", "Shanghai Ground Team", 48.0,
        "Air to land, Shanghai", True, log,
        corrective_action="refrigerant_refresh",
    )
    after = assess_order(order, model, custody_records=log)

    assert not after["sla"]["simulated_escrow_locked"]
    assert json.dumps(after, allow_nan=False)

    # A powered container's assumed buffer cannot breach the SLA.
    reefer = data[
        data.Transport_Mode.eq("Road")
        & data.Product_Category.eq("Pharmaceuticals")
        & data.Weather_Condition.eq("Hurricane")
    ].iloc[0]
    assert not assess_order(reefer, model)["sla"]["trigger_condition"]
