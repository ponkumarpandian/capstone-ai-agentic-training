"""
MediSuite Agent — CMS-1500 Claim Form PDF Generator
Generates a simplified CMS-1500 claim form using ReportLab.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


def generate_cms1500_pdf(claim_data: dict[str, Any], output_path: str | Path) -> Path:
    """Generate a CMS-1500 style claim form as a PDF.

    Args:
        claim_data: Dictionary containing claim information:
            - patient_id, name, dob, gender, address
            - provider, provider_npi, facility
            - insurance (policy_number, group_number, provider)
            - icd10_codes, cpt4_codes, amount
            - diagnoses, procedures
            - claim_id, date_of_service
        output_path: Where to save the generated PDF.

    Returns:
        Path to the generated PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ClaimTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#283593"),
        spaceBefore=12,
        spaceAfter=4,
        borderWidth=1,
        borderColor=colors.HexColor("#c5cae9"),
        borderPadding=4,
    )
    normal_style = ParagraphStyle(
        "ClaimNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )

    elements: list[Any] = []

    # ── Header ──────────────────────────────────────────────────────
    elements.append(Paragraph("HEALTH INSURANCE CLAIM FORM", title_style))
    elements.append(
        Paragraph(
            "<b>CMS-1500 (02/12)</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Claim ID: <b>{claim_data.get('claim_id', 'N/A')}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b>",
            normal_style,
        )
    )
    elements.append(Spacer(1, 12))

    # ── 1. Insurance Information ────────────────────────────────────
    elements.append(Paragraph("1. INSURANCE INFORMATION", heading_style))
    insurance = claim_data.get("insurance", {})
    ins_data = [
        ["Insurance Provider:", insurance.get("provider", claim_data.get("provider", "N/A"))],
        ["Policy Number:", insurance.get("policy_number", "N/A")],
        ["Group Number:", insurance.get("group_number", "N/A")],
        ["Subscriber Name:", insurance.get("subscriber_name", claim_data.get("name", "N/A"))],
    ]
    elements.append(_build_detail_table(ins_data))

    # ── 2. Patient Information ──────────────────────────────────────
    elements.append(Paragraph("2. PATIENT INFORMATION", heading_style))
    address = claim_data.get("address", {})
    addr_str = (
        f"{address.get('street', '')}, {address.get('city', '')}, "
        f"{address.get('state', '')} {address.get('zip', '')}"
        if address
        else "N/A"
    )
    pat_data = [
        ["Patient ID:", claim_data.get("patient_id", "N/A")],
        ["Patient Name:", claim_data.get("name", "N/A")],
        ["Date of Birth:", claim_data.get("dob", "N/A")],
        ["Gender:", claim_data.get("gender", "N/A")],
        ["Address:", addr_str],
        ["Phone:", claim_data.get("phone", "N/A")],
    ]
    elements.append(_build_detail_table(pat_data))

    # ── 3. Provider Information ─────────────────────────────────────
    elements.append(Paragraph("3. PROVIDER / FACILITY", heading_style))
    prov_data = [
        ["Provider:", claim_data.get("provider", "N/A")],
        ["NPI:", claim_data.get("provider_npi", "N/A")],
        ["Facility:", claim_data.get("facility", "N/A")],
        ["Date of Service:", claim_data.get("date_of_service", "N/A")],
    ]
    elements.append(_build_detail_table(prov_data))

    # ── 4. Diagnosis Codes (ICD-10) ─────────────────────────────────
    elements.append(Paragraph("4. DIAGNOSIS CODES (ICD-10)", heading_style))
    icd_codes = claim_data.get("icd10_codes", [])
    diagnoses = claim_data.get("diagnoses", [])
    if icd_codes:
        icd_table_data = [["#", "ICD-10 Code", "Diagnosis"]]
        for i, code in enumerate(icd_codes, 1):
            diag = diagnoses[i - 1] if i - 1 < len(diagnoses) else ""
            icd_table_data.append([str(i), code, diag])
        elements.append(_build_code_table(icd_table_data))
    else:
        elements.append(Paragraph("No diagnosis codes recorded.", normal_style))

    # ── 5. Procedure Codes (CPT-4) ──────────────────────────────────
    elements.append(Paragraph("5. PROCEDURE CODES (CPT-4)", heading_style))
    cpt_codes = claim_data.get("cpt4_codes", [])
    procedures = claim_data.get("procedures", [])
    if cpt_codes:
        cpt_table_data = [["#", "CPT-4 Code", "Procedure"]]
        for i, code in enumerate(cpt_codes, 1):
            proc = procedures[i - 1] if i - 1 < len(procedures) else ""
            cpt_table_data.append([str(i), code, proc])
        elements.append(_build_code_table(cpt_table_data))
    else:
        elements.append(Paragraph("No procedure codes recorded.", normal_style))

    # ── 6. Charges ──────────────────────────────────────────────────
    elements.append(Paragraph("6. CHARGES", heading_style))
    amount = claim_data.get("amount", 0.0)
    charge_data = [
        ["Total Charge:", f"${amount:,.2f}"],
        ["Amount Paid:", "$0.00"],
        ["Balance Due:", f"${amount:,.2f}"],
    ]
    elements.append(_build_detail_table(charge_data))

    # ── Signature Block ─────────────────────────────────────────────
    elements.append(Spacer(1, 24))
    elements.append(
        Paragraph(
            "I certify that the statements on the reverse apply to this bill and "
            "are made a part thereof.",
            normal_style,
        )
    )
    elements.append(Spacer(1, 18))
    sig_data = [
        [
            "Signed: ________________________",
            "Date: ________________________",
        ]
    ]
    sig_table = Table(sig_data, colWidths=[3.5 * inch, 3.5 * inch])
    sig_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(sig_table)

    # ── Build PDF ───────────────────────────────────────────────────
    doc.build(elements)
    logger.info("Generated CMS-1500 PDF → %s", output_path)
    return output_path


# ======================================================================
# Helper table builders
# ======================================================================

def _build_detail_table(data: list[list[str]]) -> Table:
    """Build a two-column label–value table."""
    table = Table(data, colWidths=[1.8 * inch, 5.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#37474f")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ]
        )
    )
    return table


def _build_code_table(data: list[list[str]]) -> Table:
    """Build a multi-column code table with header styling."""
    table = Table(data, colWidths=[0.5 * inch, 1.5 * inch, 5.0 * inch])
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5cae9")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    table.setStyle(TableStyle(style_cmds))
    return table
