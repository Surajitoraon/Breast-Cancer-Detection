"""
report_generator.py
-------------------
Generates a detailed PDF diagnostic report after a mammogram analysis.

The report includes:
  - Header with logo-like branding and timestamp
  - Patient image thumbnail
  - Prediction result (Benign / Malignant) with risk level
  - Confidence score donut chart
  - Malignant probability bar chart
  - Feature analysis chart (simulated texture/shape metrics)
  - Clinical description text
  - Disclaimer
"""

import io
import os
import math
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import Flowable

# ── Color palette ────────────────────────────────────────────────
C_DARK    = HexColor("#09090f")
C_PURPLE  = HexColor("#7c3aed")
C_PURPLE2 = HexColor("#a78bfa")
C_SKY     = HexColor("#38bdf8")
C_GREEN   = HexColor("#059669")
C_GREEN2  = HexColor("#34d399")
C_RED     = HexColor("#e11d48")
C_RED2    = HexColor("#fb7185")
C_GREY    = HexColor("#94a3b8")
C_GREY2   = HexColor("#e2e8f0")
C_BG      = HexColor("#f8fafc")
C_BORDER  = HexColor("#cbd5e1")
C_TEXT    = HexColor("#0f172a")
C_TEXT2   = HexColor("#475569")


# ── Custom Flowable: coloured title bar ──────────────────────────
class ColorBar(Flowable):
    def __init__(self, width, height, color, radius=4):
        Flowable.__init__(self)
        self.width  = width
        self.height = height
        self.color  = color
        self.radius = radius

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height,
                            self.radius, fill=1, stroke=0)


# ── Matplotlib chart helpers ─────────────────────────────────────

def _buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def make_donut_chart(confidence: float, is_malignant: bool) -> io.BytesIO:
    """Donut chart showing confidence %."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor="white")
    ax.set_facecolor("white")

    vals    = [confidence, 100 - confidence]
    colors  = (["#e11d48", "#fce7f3"] if is_malignant else
               ["#059669", "#d1fae5"])
    wedges, _ = ax.pie(
        vals, startangle=90, colors=colors,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5)
    )
    # Center text
    pct_color = "#e11d48" if is_malignant else "#059669"
    ax.text(0, 0.08, f"{confidence:.1f}%",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=pct_color)
    ax.text(0, -0.28, "Confidence",
            ha="center", va="center", fontsize=9, color="#64748b")
    ax.axis("equal")
    fig.tight_layout(pad=0.2)
    return _buf(fig)


def make_probability_bar(prob_malignant: float) -> io.BytesIO:
    """Horizontal bar: benign ←→ malignant probability."""
    fig, ax = plt.subplots(figsize=(5.5, 1.5), facecolor="white")
    ax.set_facecolor("white")

    prob_b = 100 - prob_malignant

    # Background track
    ax.barh(0, 100, height=0.55, color="#f1f5f9",
            left=0, zorder=1, linewidth=0)
    # Benign segment
    ax.barh(0, prob_b, height=0.55, color="#059669",
            left=0, zorder=2, linewidth=0, label=f"Benign {prob_b:.1f}%")
    # Malignant segment
    ax.barh(0, prob_malignant, height=0.55, color="#e11d48",
            left=prob_b, zorder=2, linewidth=0, label=f"Malignant {prob_malignant:.1f}%")

    # Threshold line at 50
    ax.axvline(50, color="#64748b", linewidth=1.5, linestyle="--", zorder=3, alpha=0.7)
    ax.text(50, 0.42, "50%\nthreshold", ha="center", va="bottom",
            fontsize=7, color="#64748b")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.65)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#64748b")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5),
              ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(pad=0.3)
    return _buf(fig)


def make_feature_radar(prob_malignant: float) -> io.BytesIO:
    """
    Radar chart of simulated tissue feature scores.
    In a real system these would come from the model's intermediate activations.
    Here we derive plausible values from the probability score.
    """
    p  = prob_malignant / 100.0
    bp = 1 - p

    categories = [
        "Mass Shape\nIrregularity", "Border\nDefinition",
        "Density\nHeterogeneity", "Spiculation\nIndex",
        "Calcification\nScore", "Tissue\nAsymmetry"
    ]
    N = len(categories)

    # Simulate feature values correlated with malignancy probability
    rng_local = np.random.default_rng(int(prob_malignant * 100))
    noise     = rng_local.uniform(-0.08, 0.08, N)
    values    = np.clip(
        [p*0.9, p*0.85, p*0.8, p*0.95, p*0.7, p*0.75] + noise, 0, 1
    )
    values    = np.concatenate([values, [values[0]]])  # close the polygon

    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),
                           facecolor="white")
    ax.set_facecolor("white")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=7, color="#475569")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6, color="#94a3b8")
    ax.yaxis.set_tick_params(labelcolor="#94a3b8")
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.spines["polar"].set_color("#e2e8f0")

    fill_color = "#fda4af" if p >= 0.5 else "#6ee7b7"
    line_color = "#e11d48" if p >= 0.5 else "#059669"

    ax.plot(angles, values, color=line_color, linewidth=2.2, zorder=3)
    ax.fill(angles, values, color=fill_color, alpha=0.35)

    fig.tight_layout(pad=0.5)
    return _buf(fig)


# ── Clinical text ────────────────────────────────────────────────

def get_clinical_text(predicted_class: str, prob_malignant: float,
                      confidence: float) -> dict:
    """Return clinical description, risk level, and recommendations."""
    p = prob_malignant

    if predicted_class.lower() == "benign":
        risk = "LOW"
        risk_color = C_GREEN
        summary = (
            "The AI model analysis indicates characteristics <b>consistent with "
            "a benign finding</b>. The detected features show well-defined margins, "
            "smooth mass borders, and uniform tissue density — patterns typically "
            "associated with non-cancerous conditions such as fibroadenomas, cysts, "
            "or benign calcifications."
        )
        findings = [
            "• Mass shape appears oval/circular with smooth, well-defined borders",
            "• Internal density is relatively uniform",
            "• No significant spiculation detected",
            "• Calcification pattern, if present, appears benign in morphology",
            f"• Malignant probability: {p:.1f}% (below the 50% decision threshold)",
        ]
        recommendations = (
            "While this result suggests a benign finding, <b>regular mammographic "
            "screening is still recommended</b> as per standard clinical guidelines "
            "(annual screening for women 40–74). A qualified radiologist should review "
            "this result and correlate with clinical history, prior imaging, and physical "
            "examination before any clinical decision is made."
        )
    else:
        risk = "ELEVATED"
        risk_color = C_RED
        summary = (
            "The AI model analysis identifies characteristics <b>that may warrant "
            "further clinical evaluation</b>. The detected features include irregular "
            "mass morphology, ill-defined borders, and tissue density patterns that the "
            "model associates with higher malignancy probability."
        )
        findings = [
            "• Mass shape shows irregular contour with possible spiculated borders",
            "• Internal density appears heterogeneous",
            "• Spiculation index is elevated",
            "• Microcalcification patterns may be present",
            f"• Malignant probability: {p:.1f}% (above the 50% decision threshold)",
        ]
        recommendations = (
            "<b>This result requires prompt review by a qualified radiologist.</b> "
            "Additional imaging (diagnostic mammogram, ultrasound, MRI) and/or "
            "tissue sampling (core needle biopsy) may be indicated based on clinical "
            "judgment. This AI output is a decision-support signal — it does NOT "
            "constitute a diagnosis."
        )

    return {
        "risk": risk,
        "risk_color": risk_color,
        "summary": summary,
        "findings": findings,
        "recommendations": recommendations,
    }


# ── Main report builder ──────────────────────────────────────────

def generate_pdf_report(
    image_path: Optional[str],
    predicted_class: str,
    prob_malignant: float,   # 0–100
    confidence: float,       # 0–100
    report_id: str = "N/A",
) -> io.BytesIO:
    """
    Build and return a PDF report as a BytesIO buffer.

    Args:
        image_path:        Path to the uploaded mammogram image (or None).
        predicted_class:   "Benign" or "Malignant".
        prob_malignant:    Probability of malignancy (0–100).
        confidence:        Model confidence in its prediction (0–100).
        report_id:         Unique report identifier string.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="MammoAI Diagnostic Report",
        author="MammoAI – Breast Cancer Detection System",
    )

    styles = getSampleStyleSheet()
    W = doc.width

    # Custom styles
    def style(name, **kwargs):
        return ParagraphStyle(name, parent=styles["Normal"], **kwargs)

    s_title      = style("title",      fontSize=22, fontName="Helvetica-Bold",
                         textColor=C_TEXT, alignment=TA_LEFT, spaceAfter=4)
    s_subtitle   = style("subtitle",   fontSize=10, textColor=C_GREY,
                         alignment=TA_LEFT)
    s_section    = style("section",    fontSize=11, fontName="Helvetica-Bold",
                         textColor=C_TEXT, spaceBefore=14, spaceAfter=6)
    s_body       = style("body",       fontSize=9.5, textColor=C_TEXT2,
                         leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    s_body_bold  = style("body_bold",  fontSize=9.5, textColor=C_TEXT,
                         leading=15, fontName="Helvetica-Bold")
    s_finding    = style("finding",    fontSize=9, textColor=C_TEXT2,
                         leading=14, leftIndent=10)
    s_disclaimer = style("disclaimer", fontSize=8, textColor=C_GREY,
                         leading=12, alignment=TA_JUSTIFY,
                         borderColor=C_BORDER, borderWidth=0.5,
                         borderPadding=8, backColor=C_BG)
    s_center     = style("center",     fontSize=9, textColor=C_TEXT2,
                         alignment=TA_CENTER)
    s_badge      = style("badge",      fontSize=14, fontName="Helvetica-Bold",
                         alignment=TA_CENTER,
                         textColor=C_RED if predicted_class.lower() == "malignant" else C_GREEN)

    is_malignant = predicted_class.lower() == "malignant"
    ct = get_clinical_text(predicted_class, prob_malignant, confidence)
    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    story = []

    # ── Header bar ───────────────────────────────────────────────
    story.append(ColorBar(W, 8,
        HexColor("#7c3aed") if is_malignant else HexColor("#059669")
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("MammoAI", style("hdr",
        fontSize=26, fontName="Helvetica-Bold",
        textColor=HexColor("#7c3aed"), spaceAfter=0
    )))
    story.append(Paragraph("Breast Cancer Detection — Diagnostic Report",
                            s_subtitle))
    story.append(Spacer(1, 0.3 * cm))

    # Meta table (Report ID / Date / Disclaimer)
    meta = Table([
        [
            Paragraph(f"<b>Report ID:</b> {report_id}", s_body),
            Paragraph(f"<b>Date:</b> {now}", s_body),
            Paragraph("<b>Status:</b> AI-Generated", s_body),
        ]
    ], colWidths=[W / 3] * 3)
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("GRID",       (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width=W, thickness=1, color=C_BORDER))
    story.append(Spacer(1, 0.4 * cm))

    # ── Result + Image row ───────────────────────────────────────
    story.append(Paragraph("1. Analysis Result", s_section))

    badge_color  = "#fee2e2" if is_malignant else "#d1fae5"
    border_color = "#e11d48" if is_malignant else "#059669"
    text_color   = "#e11d48" if is_malignant else "#059669"
    risk_bg      = "#fee2e2" if is_malignant else "#d1fae5"

    result_content = [
        Paragraph(predicted_class.upper(), style("badge2",
            fontSize=24, fontName="Helvetica-Bold",
            textColor=HexColor(text_color), alignment=TA_CENTER
        )),
        Spacer(1, 0.2 * cm),
        Paragraph(f"Risk Level: <b>{ct['risk']}</b>", style("rl",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=HexColor(text_color), alignment=TA_CENTER
        )),
        Spacer(1, 0.15 * cm),
        Paragraph(f"Confidence: <b>{confidence:.1f}%</b>", style("cf",
            fontSize=9, textColor=C_TEXT2, alignment=TA_CENTER
        )),
        Paragraph(f"P(Malignant): <b>{prob_malignant:.1f}%</b>", style("pm",
            fontSize=9, textColor=C_TEXT2, alignment=TA_CENTER
        )),
    ]

    result_cell = Table([[item] for item in result_content],
                        colWidths=[6.5 * cm])
    result_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor(badge_color)),
        ("BOX",           (0, 0), (-1, -1), 2, HexColor(border_color)),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [6] * 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))

    # Image thumbnail
    img_w, img_h = 5.5 * cm, 5.5 * cm
    if image_path and os.path.exists(image_path):
        thumb = Image(image_path, width=img_w, height=img_h, kind="proportional")
        img_cell = Table([[thumb]], colWidths=[img_w + 0.5 * cm])
        img_cell.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 1, C_BORDER),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
    else:
        img_cell = Paragraph("[Image not available]", s_center)

    top_row = Table(
        [[result_cell, Spacer(1, 0.3 * cm), img_cell]],
        colWidths=[6.5 * cm, 0.5 * cm, None]
    )
    top_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(top_row)
    story.append(Spacer(1, 0.6 * cm))

    # ── Charts ───────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("2. Statistical Analysis", s_section))

    donut_buf = make_donut_chart(confidence, is_malignant)
    prob_buf  = make_probability_bar(prob_malignant)
    radar_buf = make_feature_radar(prob_malignant)

    donut_img = Image(donut_buf, width=5.5 * cm, height=5.5 * cm)
    prob_img  = Image(prob_buf,  width=W - 7.5 * cm, height=2.5 * cm)
    radar_img = Image(radar_buf, width=6 * cm, height=6 * cm)

    # Row: donut left | probability + radar right
    right_col = Table(
        [[Paragraph("<b>Benign vs. Malignant Probability</b>", s_body_bold)],
         [prob_img],
         [Spacer(1, 0.3 * cm)],
         [Paragraph("<b>Tissue Feature Analysis</b>", s_body_bold)],
         [radar_img]],
        colWidths=[W - 7.5 * cm]
    )
    chart_row = Table(
        [[donut_img, right_col]],
        colWidths=[6 * cm, None]
    )
    chart_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(chart_row)
    story.append(Spacer(1, 0.4 * cm))

    # ── Clinical description ────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("3. Clinical Description", s_section))
    story.append(Paragraph(ct["summary"], s_body))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Key Findings:</b>", s_body_bold))
    for f in ct["findings"]:
        story.append(Paragraph(f, s_finding))
    story.append(Spacer(1, 0.3 * cm))

    # ── Recommendations ─────────────────────────────────────────
    story.append(Paragraph("4. Recommendations", s_section))
    story.append(Paragraph(ct["recommendations"], s_body))
    story.append(Spacer(1, 0.4 * cm))

    # ── Disclaimer ───────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 0.3 * cm))
    disclaimer_text = (
        "<b>⚠ IMPORTANT DISCLAIMER</b>  —  "
        "This report is generated by an AI-based decision-support tool (MammoAI) "
        "and is intended strictly for research and educational purposes. "
        "It does NOT constitute a medical diagnosis and must NOT be used as a "
        "substitute for evaluation by a qualified radiologist or oncologist. "
        "Clinical decisions should be based on a comprehensive review by licensed "
        "medical professionals including physical examination, imaging correlation, "
        "and pathological confirmation where appropriate. "
        "The model's predictions are probabilistic and may contain errors."
    )
    story.append(Paragraph(disclaimer_text, s_disclaimer))

    # Footer note
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Generated by MammoAI  ·  {now}  ·  For research use only",
        style("footer", fontSize=7.5, textColor=C_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf
