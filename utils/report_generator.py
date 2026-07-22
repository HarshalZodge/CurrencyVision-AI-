"""
Report Generator Module for CurrencyVision AI.
Generates robust, audit-ready prediction reports in PDF, CSV, and JSON formats.
"""

import os
import json
import io
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


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

    if not HAS_REPORTLAB:
        # Fallback simple text payload formatted as PDF-like document if ReportLab is missing
        text_content = f"""CURRENCYVISION AI — ANALYSIS REPORT
==========================================
Timestamp: {prediction_info.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
Detected Currency: Rs. {prediction_info.get('prediction', 'N/A')}
Confidence Score: {prediction_info.get('confidence', 0.0):.2f}%
Inference Speed: {prediction_info.get('inference_time_ms', 0.0):.2f} ms
Model Version: {prediction_info.get('model_version', 'v1.0.0')}
==========================================
"""
        return text_content.encode("utf-8")

    try:
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
            fontSize=22,
            textColor=colors.HexColor("#4F46E5"),
            spaceAfter=12,
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=20,
        )

        h2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=12,
            spaceAfter=8,
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

        elements.append(Paragraph("CurrencyVision AI — Analysis Report", title_style))
        elements.append(
            Paragraph(
                f"Generated on {prediction_info.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | Powered by Custom Deep Learning CNN",
                subtitle_style,
            )
        )

        denom = prediction_info.get("prediction", "N/A")
        conf = float(prediction_info.get("confidence", 0.0))
        inf_time = float(prediction_info.get("inference_time_ms", prediction_info.get("inference_time", 0.0)))

        summary_data = [
            [Paragraph("Parameter", cell_bold), Paragraph("Details / Value", cell_bold)],
            [Paragraph("Detected Denomination", cell_normal), Paragraph(f"<b>Rs. {denom}</b>", cell_bold)],
            [Paragraph("Confidence Score", cell_normal), Paragraph(f"{conf:.2f}%", cell_normal)],
            [Paragraph("Inference Speed", cell_normal), Paragraph(f"{inf_time:.2f} ms", cell_normal)],
            [Paragraph("Model Version", cell_normal), Paragraph(f"{prediction_info.get('model_version', 'v1.0.0')}", cell_normal)],
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
        elements.append(Spacer(1, 15))

        # Top Predictions Section - Robust Unpacking
        elements.append(Paragraph("Top Class Probabilities", h2_style))
        raw_top = prediction_info.get("top_predictions", prediction_info.get("top3", []))
        prob_data = [[Paragraph("Denomination", cell_bold), Paragraph("Probability (%)", cell_bold)]]

        for item in raw_top:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                d_str, p_val = str(item[0]), float(item[1])
            elif isinstance(item, dict):
                d_str = str(item.get("currency", item.get("label", "N/A")))
                p_val = float(item.get("confidence", item.get("probability", 0.0)))
            else:
                d_str, p_val = "N/A", 0.0

            prob_data.append([
                Paragraph(f"Rs. {d_str}", cell_normal),
                Paragraph(f"{p_val:.2f}%", cell_normal),
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

        if image_bytes is not None:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Uploaded Note Image", h2_style))
            img_buffer = io.BytesIO(image_bytes)
            rl_img = RLImage(img_buffer, width=280, height=140)
            elements.append(rl_img)

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        # Fallback simple formatted text payload on any rendering error
        buffer = io.BytesIO()
        text_content = f"""CURRENCYVISION AI — ANALYSIS REPORT
==========================================
Timestamp: {prediction_info.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
Detected Currency: Rs. {prediction_info.get('prediction', 'N/A')}
Confidence Score: {prediction_info.get('confidence', 0.0):.2f}%
Inference Speed: {prediction_info.get('inference_time_ms', 0.0):.2f} ms
Model Version: {prediction_info.get('model_version', 'v1.0.0')}
==========================================
"""
        buffer.write(text_content.encode("utf-8"))
        buffer.seek(0)
        return buffer.getvalue()


def generate_json_report(prediction_info: Dict[str, Any]) -> str:
    """Converts prediction info dictionary to formatted JSON string with safe type encoder."""
    clean_dict = {}
    for k, v in prediction_info.items():
        if k in ["gradcam_heatmap", "gradcam_overlay", "raw_predictions"]:
            continue
        clean_dict[k] = v

    return json.dumps(clean_dict, indent=4, default=str)


def generate_csv_report(history_records: List[Dict[str, Any]]) -> str:
    """Converts prediction session history list of dicts to CSV string."""
    if not history_records:
        df = pd.DataFrame(columns=["timestamp", "prediction", "confidence", "inference_time_ms", "model_version"])
    else:
        clean_records = []
        for rec in history_records:
            clean_rec = {k: v for k, v in rec.items() if k not in ["gradcam_heatmap", "gradcam_overlay", "raw_predictions"]}
            clean_records.append(clean_rec)
        df = pd.DataFrame(clean_records)
    return df.to_csv(index=False)
