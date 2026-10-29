
import re


def extract_claim_details(text: str) -> dict:
    """Extract and validate basic insurance claim details."""

    claim_id_match = re.search(
        r"Claim\s*ID\s*:?\s*([A-Za-z0-9-]+)",
        text,
        re.IGNORECASE
    )

    amount_match = re.search(
        r"Claim\s*Amount\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE
    )

    incident_match = re.search(
        r"Incident\s*Type\s*:?\s*([^\r\n]+)",
        text,
        re.IGNORECASE
    )

    claim_id = claim_id_match.group(1) if claim_id_match else None

    claim_amount = (
        float(amount_match.group(1).replace(",", ""))
        if amount_match else None
    )

    incident_type = (
        incident_match.group(1).strip()
        if incident_match else None
    )

    errors = []

    if not claim_id:
        errors.append("Claim ID is missing")

    if claim_amount is None:
        errors.append("Claim amount is missing")
    elif claim_amount <= 0:
        errors.append("Claim amount must be greater than zero")

    if not incident_type:
        errors.append("Incident type is missing")

    return {
        "claim_id": claim_id,
        "claim_amount": claim_amount,
        "incident_type": incident_type,
        "is_valid": len(errors) == 0,
        "validation_errors": errors
    }
