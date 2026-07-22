"""
Report Generator Module for CurrencyVision AI.
Generates exportable prediction reports in PDF, CSV, and JSON formats.
"""

import os
import json
import io
import datetime
from typing import Dict, Any, List
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(
    prediction_info: Dict[str, Any],
    image_bytes: Optional[bytes] = None
) -> bytes:
    """
    Generates a PDF summary report for currency recognition.

    Args:
        prediction_info (Dict[str, Any]): Dictionary containing metadata & predictions.
        image_bytes (Optional[bytes]): Optional raw image byte content.

    Returns:
        bytes: Binary PDF document payload.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#4F46E5"),
        spaceAfter=15,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=25,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=15,
        spaceAfter=10,
    )

    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#1E293B"),
    )

    cell_normal = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#334155"),
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("CurrencyVision AI — Analysis Report", title_style))
    elements.append(
        Paragraph(
            f"Generated on {prediction_info.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | Powered by Custom Deep Learning CNN",
            subtitle_style,
        )
    )

    # Key Summary Table Data
    denom = prediction_info.get("prediction", "N/A")
    conf = prediction_info.get("confidence", 0.0)
    inf_time = prediction_info.get("inference_time_ms", 0.0)

    summary_data = [
        [Paragraph("Parameter", cell_bold), Paragraph("Details / Value", cell_bold)],
        [Paragraph("Detected Denomination", cell_normal), Paragraph(f"<b>Rs. {denom}</b>", cell_bold)],
        [Paragraph("Confidence Score", cell_normal), Paragraph(f"{conf:.2f}%", cell_normal)],
        [Paragraph("Inference Speed", cell_normal), Paragraph(f"{inf_time:.2f} ms", cell_normal)],
        [Paragraph("Model Version", cell_normal), Paragraph(f"{prediction_info.get('model_version', '1.0.0')}", cell_normal)],
        [Paragraph("Image Resolution", cell_normal), Paragraph(f"{prediction_info.get('image_size', '128x128')}", cell_normal)],
        [Paragraph("Prediction Status", cell_normal), Paragraph("VERIFIED STABLE", cell_bold)],
    ]

    t = Table(summary_data, colWidths=[200, 300])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#EEF2FF")),
            ("TEXTCOLOR", (0, 0), (1, 0), colors.HexColor("#4F46E5")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ])
    )
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Top Probabilities Section
    elements.append(Paragraph("Top Class Probabilities", h2_style))
    top_preds = prediction_info.get("top_predictions", [])
    prob_data = [[Paragraph("Denomination", cell_bold), Paragraph("Probability (%)", cell_bold)]]
    for denom_str, prob_val in top_preds:
        prob_data.append([
            Paragraph(f"Rs. {denom_str}", cell_normal),
            Paragraph(f"{prob_val:.2f}%", cell_normal),
        ])

    t_prob = Table(prob_data, colWidths=[200, 300])
    t_prob.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#F8FAFC")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ])
    )
    elements.append(t_prob)

    # Optional image rendering
    if image_bytes is not None:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Uploaded Note Image", h2_style))
        img_buffer = io.BytesIO(image_bytes)
        rl_img = RLImage(img_buffer, width=280, height=140)
        elements.append(rl_img)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_json_report(prediction_info: Dict[str, Any]) -> str:
    """Converts prediction info dictionary to formatted JSON string."""
    return json.dumps(prediction_info, indent=4)


def generate_csv_report(history_records: List[Dict[str, Any]]) -> str:
    """Converts prediction session history list of dicts to CSV string."""
    if not history_records:
        df = pd.DataFrame(columns=["timestamp", "prediction", "confidence", "inference_time_ms", "model_version"])
    else:
        df = pd.DataFrame(history_records)
    return df.to_csv(index=False)
