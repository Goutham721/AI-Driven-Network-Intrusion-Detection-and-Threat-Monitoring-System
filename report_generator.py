"""
report_generator.py
Generates a PDF security report summarizing simulated traffic analysis,
attack statistics, model performance, and recent security events.
Uses fpdf2 (lightweight, pure-Python, no external system dependencies --
a reliable choice for a Windows student environment).
"""

import os
import sys
from datetime import datetime

from fpdf import FPDF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR, PDF_REPORT_PATH  # noqa: E402
from src.utils import safe_makedirs  # noqa: E402


class SecurityReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "AI-Driven Network Intrusion Detection System", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, "Security Report (Simulated Traffic Analysis)", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)


def generate_pdf_report(summary_stats: dict, model_metrics: dict,
                         recent_events: list) -> str:
    """
    Build the PDF report and save it to outputs/intrusion_report.pdf.

    summary_stats: dict with keys like total_traffic, normal_traffic,
        attack_traffic, attack_percentage, attack_distribution (dict),
        severity_distribution (dict)
    model_metrics: dict with keys model_name, accuracy, precision,
        recall, f1_score
    recent_events: list of dict rows (from the intrusion log)
    """
    safe_makedirs(OUTPUTS_DIR)

    pdf = SecurityReportPDF()
    pdf.add_page()

    pdf.body_text(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(2)

    # --- Traffic summary ---
    pdf.section_title("Traffic Summary")
    pdf.body_text(
        f"Total traffic analyzed: {summary_stats.get('total_traffic', 0)}\n"
        f"Normal traffic: {summary_stats.get('normal_traffic', 0)}\n"
        f"Attack traffic: {summary_stats.get('attack_traffic', 0)}\n"
        f"Attack percentage: {summary_stats.get('attack_percentage', 0):.2f}%"
    )

    # --- Attack type distribution ---
    pdf.section_title("Attack Type Distribution")
    attack_dist = summary_stats.get("attack_distribution", {})
    if attack_dist:
        lines = "\n".join(f"- {k}: {v}" for k, v in attack_dist.items())
        pdf.body_text(lines)
    else:
        pdf.body_text("No attacks recorded in this session.")

    # --- Severity distribution ---
    pdf.section_title("Severity Distribution")
    sev_dist = summary_stats.get("severity_distribution", {})
    if sev_dist:
        lines = "\n".join(f"- {k}: {v}" for k, v in sev_dist.items())
        pdf.body_text(lines)
    else:
        pdf.body_text("No severity data available.")

    # --- Model performance ---
    pdf.section_title("Model Performance")
    pdf.body_text(
        f"Model used: {model_metrics.get('model_name', 'N/A')}\n"
        f"Accuracy: {model_metrics.get('accuracy', 0):.4f}\n"
        f"Precision: {model_metrics.get('precision', 0):.4f}\n"
        f"Recall: {model_metrics.get('recall', 0):.4f}\n"
        f"F1-score: {model_metrics.get('f1_score', 0):.4f}"
    )

    # --- Recent events ---
    pdf.section_title("Recent Security Events")
    if recent_events:
        pdf.set_font("Helvetica", "B", 8)
        headers = ["Time", "Source IP", "Attack", "Severity", "Action"]
        widths = [30, 35, 35, 25, 45]
        for h, w in zip(headers, widths):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for event in recent_events[:20]:
            row = [
                str(event.get("timestamp", ""))[:19],
                str(event.get("source_ip", ""))[:18],
                str(event.get("attack_type", ""))[:18],
                str(event.get("severity", "")),
                str(event.get("action", "")),
            ]
            for val, w in zip(row, widths):
                pdf.cell(w, 6, val, border=1)
            pdf.ln()
    else:
        pdf.body_text("No security events recorded in this session.")

    # --- Recommendations ---
    pdf.section_title("Recommendations")
    pdf.body_text(
        "- Continue monitoring flagged source IPs for repeated activity.\n"
        "- Review CRITICAL and HIGH severity events promptly.\n"
        "- Periodically retrain the model on updated traffic patterns.\n"
        "- This report reflects a software-based simulation and should be "
        "used for academic demonstration purposes only."
    )

    pdf.output(PDF_REPORT_PATH)
    return PDF_REPORT_PATH
