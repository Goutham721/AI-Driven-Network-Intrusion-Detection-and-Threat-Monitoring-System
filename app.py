"""
app.py
Main Streamlit dashboard for the AI-Driven Network Intrusion Detection
and Threat Monitoring System. All traffic shown here is SIMULATED.
"""

import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    ATTACK_PROBABILITY,
    DEFAULT_SIMULATION_SPEED_SEC,
    METRICS_PATH,
    MODELS_DIR,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
)
from src.alert_manager import append_to_log, create_alert_entry, load_log, filter_log, save_log
from src.predict import predict_record
from src.report_generator import generate_pdf_report
from src.threat_analyzer import severity_icon
from src.traffic_simulator import generate_live_record

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "log_df" not in st.session_state:
    st.session_state.log_df = load_log()
if "event_counter" not in st.session_state:
    st.session_state.event_counter = len(st.session_state.log_df)
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "live_records" not in st.session_state:
    st.session_state.live_records = []
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_normal" not in st.session_state:
    st.session_state.total_normal = 0
if "total_attacks" not in st.session_state:
    st.session_state.total_attacks = 0
if "confidence_sum" not in st.session_state:
    st.session_state.confidence_sum = 0.0


def models_available() -> bool:
    return os.path.exists(STAGE1_MODEL_PATH) and os.path.exists(STAGE2_MODEL_PATH)


def process_one_record(attack_probability: float):
    """Generate one simulated record, run it through the pipeline, and log it."""
    record = generate_live_record(attack_probability)
    try:
        prediction = predict_record(record)
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state.simulation_running = False
        return

    st.session_state.total_processed += 1
    st.session_state.confidence_sum += prediction["confidence"]

    if prediction["status"] == "NORMAL":
        st.session_state.total_normal += 1
    else:
        st.session_state.total_attacks += 1
        st.session_state.event_counter += 1
        entry = create_alert_entry(record, prediction, st.session_state.event_counter)
        st.session_state.log_df = append_to_log(st.session_state.log_df, entry)
        save_log(st.session_state.log_df)

    display_row = {
        "Time": time.strftime("%H:%M:%S"),
        "Source IP": record["source_ip"],
        "Destination IP": record["destination_ip"],
        "Protocol": record["protocol"],
        "Packets": record["packet_count"],
        "Status": prediction["status"],
        "Attack": prediction["attack_type"] or "-",
        "Severity": prediction["severity"],
    }
    st.session_state.live_records.insert(0, display_row)
    st.session_state.live_records = st.session_state.live_records[:200]

    return prediction, record


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Live Traffic Simulation",
        "Intrusion Detection",
        "Threat Monitoring",
        "Attack Analysis",
        "Model Performance",
        "Intrusion Logs",
        "Reports",
        "About Project",
    ],
)

if not models_available():
    st.sidebar.warning(
        "⚠️ Models not found. Run:\n\n"
        "1. `python src/data_generator.py`\n"
        "2. `python src/train_model.py`\n\n"
        "before using detection features."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("AI-Driven Network Intrusion Detection and Threat Monitoring System")
st.caption("AI-Based Software Simulation for Network Security Monitoring")

# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    total = st.session_state.total_processed
    normal = st.session_state.total_normal
    attacks = st.session_state.total_attacks
    attack_pct = (attacks / total * 100) if total else 0.0
    avg_conf = (st.session_state.confidence_sum / total) if total else 0.0

    log_df = st.session_state.log_df
    critical_count = (log_df["severity"] == "CRITICAL").sum() if not log_df.empty else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Traffic Records", total)
    c2.metric("Normal Traffic", normal)
    c3.metric("Detected Attacks", attacks)
    c4.metric("Attack Percentage", f"{attack_pct:.1f}%")
    c5.metric("Critical Threats", int(critical_count))
    c6.metric("Average Confidence", f"{avg_conf:.2f}")

    st.info(
        "Go to **Live Traffic Simulation** to start generating simulated traffic "
        "and see this dashboard populate in real time."
    )

    if not log_df.empty:
        st.subheader("Recent Alerts")
        st.dataframe(log_df.tail(10).iloc[::-1], use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Live Traffic Simulation
# ---------------------------------------------------------------------------
elif page == "Live Traffic Simulation":
    st.subheader("🔴 Live Simulation — *Simulated Network Traffic*")

    col1, col2, col3, col4 = st.columns(4)
    num_records = col1.number_input("Records to generate", min_value=1, max_value=500, value=20)
    speed = col2.slider("Simulation speed (sec/record)", 0.0, 2.0, DEFAULT_SIMULATION_SPEED_SEC)
    attack_prob = col3.slider("Attack probability", 0.0, 1.0, ATTACK_PROBABILITY)

    start_col, stop_col = col4.columns(2)
    if start_col.button("▶ Start", disabled=not models_available()):
        st.session_state.simulation_running = True
    if stop_col.button("⏹ Stop"):
        st.session_state.simulation_running = False

    table_placeholder = st.empty()
    kpi_placeholder = st.empty()

    if st.session_state.simulation_running:
        for _ in range(int(num_records)):
            if not st.session_state.simulation_running:
                break
            process_one_record(attack_prob)

            with kpi_placeholder.container():
                k1, k2, k3 = st.columns(3)
                k1.metric("Processed this run", len(st.session_state.live_records))
                k2.metric("Total Normal", st.session_state.total_normal)
                k3.metric("Total Attacks", st.session_state.total_attacks)

            with table_placeholder.container():
                st.dataframe(
                    pd.DataFrame(st.session_state.live_records[:50]),
                    use_container_width=True,
                )
            time.sleep(speed)
        st.session_state.simulation_running = False
        st.success("Simulation batch complete.")
    else:
        if st.session_state.live_records:
            table_placeholder.dataframe(
                pd.DataFrame(st.session_state.live_records[:50]), use_container_width=True
            )
        else:
            st.write("No simulated traffic yet. Click **Start** to begin.")

# ---------------------------------------------------------------------------
# Page: Intrusion Detection (single record test)
# ---------------------------------------------------------------------------
elif page == "Intrusion Detection":
    st.subheader("Manual Traffic Record Test")
    st.write("Enter traffic feature values manually to test the detection pipeline.")

    if not models_available():
        st.warning("Train the models first (see sidebar instructions).")
    else:
        with st.form("manual_test_form"):
            c1, c2, c3 = st.columns(3)
            source_port = c1.number_input("Source Port", 1, 65535, 51000)
            destination_port = c1.number_input("Destination Port", 1, 65535, 443)
            protocol = c1.selectbox("Protocol", ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"])
            tcp_flags = c1.selectbox("TCP Flags", ["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK"])

            packet_count = c2.number_input("Packet Count", 0, 100000, 150)
            byte_count = c2.number_input("Byte Count", 0, 10000000, 20000)
            packet_size = c2.number_input("Packet Size", 0, 5000, 500)
            duration = c2.number_input("Duration (sec)", 0.0, 500.0, 5.0)

            pps = c3.number_input("Packets/sec", 0.0, 10000.0, 10.0)
            bps = c3.number_input("Bytes/sec", 0.0, 1000000.0, 2000.0)
            conn_count = c3.number_input("Connection Count", 0, 1000, 2)
            failed_conn = c3.number_input("Failed Connections", 0, 1000, 0)

            c4, c5 = st.columns(2)
            request_rate = c4.number_input("Request Rate", 0.0, 5000.0, 5.0)
            response_rate = c4.number_input("Response Rate", 0.0, 5000.0, 5.0)
            payload_size = c5.number_input("Payload Size", 0, 5000, 500)
            flow_rate = c5.number_input("Flow Rate", 0.0, 5000.0, 5.0)

            submitted = st.form_submit_button("Run Detection")

        if submitted:
            record = {
                "source_ip": "203.0.113.10",
                "destination_ip": "10.0.0.5",
                "source_port": source_port,
                "destination_port": destination_port,
                "protocol": protocol,
                "tcp_flags": tcp_flags,
                "packet_count": packet_count,
                "byte_count": byte_count,
                "packet_size": packet_size,
                "duration": duration,
                "packets_per_second": pps,
                "bytes_per_second": bps,
                "connection_count": conn_count,
                "failed_connections": failed_conn,
                "request_rate": request_rate,
                "response_rate": response_rate,
                "payload_size": payload_size,
                "flow_rate": flow_rate,
            }
            result = predict_record(record)

            if result["status"] == "NORMAL":
                st.success(f"✅ NORMAL traffic (confidence: {result['confidence']:.2f})")
            else:
                st.error(
                    f"🚨 ATTACK DETECTED: {result['attack_type']} "
                    f"(confidence: {result['confidence']:.2f}, severity: {result['severity']})"
                )
                st.write("**Possible contributing indicators:**")
                for ind in result["contributing_indicators"]:
                    st.write(f"- {ind}")
                st.caption(
                    "These are rule-based heuristic indicators, not SHAP explanations."
                )

# ---------------------------------------------------------------------------
# Page: Threat Monitoring
# ---------------------------------------------------------------------------
elif page == "Threat Monitoring":
    st.subheader("Network Threat Monitoring")

    log_df = st.session_state.log_df
    if log_df.empty:
        st.write("🟢 **NORMAL** — No threats detected yet. Run the live simulation to generate events.")
    else:
        latest = log_df.iloc[-1]
        st.markdown(f"### {severity_icon(latest['severity'])} Current Status: {latest['severity']}")
        st.write("**Latest detected threat:**")
        st.json(latest.to_dict())

        st.subheader("Recent Threats")
        st.dataframe(log_df.tail(15).iloc[::-1], use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Attack Analysis
# ---------------------------------------------------------------------------
elif page == "Attack Analysis":
    st.subheader("Attack & Traffic Visualizations")

    log_df = st.session_state.log_df
    live_df = pd.DataFrame(st.session_state.live_records)

    if log_df.empty and live_df.empty:
        st.write("No data yet — run the Live Traffic Simulation first.")
    else:
        col1, col2 = st.columns(2)

        if not live_df.empty:
            with col1:
                status_counts = live_df["Status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig = px.pie(status_counts, names="Status", values="Count",
                             title="Normal vs Attack Traffic")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                proto_counts = live_df["Protocol"].value_counts().reset_index()
                proto_counts.columns = ["Protocol", "Count"]
                fig = px.bar(proto_counts, x="Protocol", y="Count", title="Protocol Distribution")
                st.plotly_chart(fig, use_container_width=True)

        if not log_df.empty:
            col3, col4 = st.columns(2)
            with col3:
                attack_counts = log_df["attack_type"].value_counts().reset_index()
                attack_counts.columns = ["Attack Type", "Count"]
                fig = px.bar(attack_counts, x="Attack Type", y="Count",
                             title="Attack Type Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                sev_counts = log_df["severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig = px.pie(sev_counts, names="Severity", values="Count",
                             title="Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)

            col5, col6 = st.columns(2)
            with col5:
                top_ips = log_df["source_ip"].value_counts().head(10).reset_index()
                top_ips.columns = ["Source IP", "Count"]
                fig = px.bar(top_ips, x="Source IP", y="Count", title="Top Source IPs")
                st.plotly_chart(fig, use_container_width=True)

            with col6:
                top_ports = log_df["destination_port"].value_counts().head(10).reset_index()
                top_ports.columns = ["Destination Port", "Count"]
                fig = px.bar(top_ports, x="Destination Port", y="Count",
                             title="Top Destination Ports")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance")

    if not os.path.exists(METRICS_PATH):
        st.warning("No metrics found. Run `python src/train_model.py` first.")
    else:
        import json
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        for stage_key, stage_title in [("stage1", "Stage 1: Normal vs Attack"),
                                        ("stage2", "Stage 2: Attack Type Classification")]:
            st.markdown(f"### {stage_title}")
            stage = metrics[stage_key]
            st.write(f"**Best model:** {stage['best_model']}")

            comp_df = pd.DataFrame(stage["comparison"]).T
            st.dataframe(comp_df, use_container_width=True)

            cm = stage["confusion_matrix"]
            labels = stage["labels"]
            fig = px.imshow(
                cm, x=labels, y=labels, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title=f"{stage_title} — Confusion Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

        if metrics.get("feature_importance"):
            st.markdown("### Feature Importance")
            for stage_key, importances in metrics["feature_importance"].items():
                imp_df = pd.DataFrame(
                    sorted(importances.items(), key=lambda x: -x[1]),
                    columns=["Feature", "Importance"],
                )
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             title=f"Feature Importance ({stage_key})")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Intrusion Logs
# ---------------------------------------------------------------------------
elif page == "Intrusion Logs":
    st.subheader("Intrusion Event Log")

    log_df = st.session_state.log_df
    if log_df.empty:
        st.write("No events logged yet.")
    else:
        c1, c2, c3 = st.columns(3)
        attack_filter = c1.selectbox("Attack Type", ["All"] + sorted(log_df["attack_type"].unique().tolist()))
        severity_filter = c2.selectbox("Severity", ["All"] + sorted(log_df["severity"].unique().tolist()))
        ip_filter = c3.text_input("Source IP contains")

        filtered = filter_log(log_df, attack_filter, severity_filter, ip_filter)
        st.dataframe(filtered.iloc[::-1], use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Log as CSV", csv, "intrusion_logs.csv", "text/csv")

# ---------------------------------------------------------------------------
# Page: Reports
# ---------------------------------------------------------------------------
elif page == "Reports":
    st.subheader("Security Report")

    if st.button("Generate PDF Report"):
        total = st.session_state.total_processed
        normal = st.session_state.total_normal
        attacks = st.session_state.total_attacks
        attack_pct = (attacks / total * 100) if total else 0.0

        log_df = st.session_state.log_df
        attack_dist = log_df["attack_type"].value_counts().to_dict() if not log_df.empty else {}
        sev_dist = log_df["severity"].value_counts().to_dict() if not log_df.empty else {}

        summary_stats = {
            "total_traffic": total,
            "normal_traffic": normal,
            "attack_traffic": attacks,
            "attack_percentage": attack_pct,
            "attack_distribution": attack_dist,
            "severity_distribution": sev_dist,
        }

        model_metrics = {"model_name": "N/A", "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        if os.path.exists(METRICS_PATH):
            import json
            with open(METRICS_PATH) as f:
                m = json.load(f)
            best = m["stage1"]["best_model"]
            comp = m["stage1"]["comparison"][best]
            model_metrics = {
                "model_name": best,
                "accuracy": comp["accuracy"],
                "precision": comp["precision"],
                "recall": comp["recall"],
                "f1_score": comp["f1_score"],
            }

        recent_events = log_df.tail(20).to_dict("records") if not log_df.empty else []

        try:
            path = generate_pdf_report(summary_stats, model_metrics, recent_events)
            st.success(f"Report generated: {path}")
            with open(path, "rb") as f:
                st.download_button("⬇ Download PDF Report", f, "intrusion_report.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

# ---------------------------------------------------------------------------
# Page: About Project
# ---------------------------------------------------------------------------
elif page == "About Project":
    st.subheader("About This Project")
    st.markdown(
        """
**AI-Driven Network Intrusion Detection and Threat Monitoring System**

A software-only, simulation-based academic project demonstrating how
machine learning can be applied to network intrusion detection.

- **No physical hardware required** — runs entirely on a Windows PC.
- **All traffic is synthetically generated** — no real network scanning,
  no real attacks, no real traffic capture.
- Uses a **two-stage ML pipeline**: binary Normal/Attack detection
  followed by multiclass attack-type classification.
- Security responses shown in this dashboard (MONITOR, ALERT, FLAG SOURCE,
  SIMULATED BLOCK, ESCALATE) are **simulated only** and never act on real
  network connections.

Built with Python, scikit-learn, Streamlit, and Plotly.
        """
    )

