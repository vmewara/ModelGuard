from pathlib import Path
from datetime import datetime, date
import argparse
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SIMULATED_DIR = DATA_DIR / "simulated"
MODEL_DIR = BASE_DIR

REPORTS = {
    "PSI Report": "modelguard_psi_report.csv",
    "KS Report": "modelguard_ks_report.csv",
    "Data Quality": "modelguard_data_quality_report.csv",
    "Alerts": "modelguard_alerts.csv",
    "Monitoring History": "modelguard_monitoring_history.csv",
}

REFERENCE_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

REFERENCE_CANDIDATES = [
    BASE_DIR / "reference_data.csv",
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    BASE_DIR / "telco_customer_churn.csv",
    BASE_DIR / "Telco-Customer-Churn.csv",
    DATA_DIR / "reference_data.csv",
    DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    DATA_DIR / "telco_customer_churn.csv",
    DATA_DIR / "Telco-Customer-Churn.csv",
]

PSI_LOW = 0.10
PSI_HIGH = 0.25


# =====================================================
# SAFE FILE FUNCTIONS
# =====================================================
def read_csv_safe(path):
    try:
        path = Path(path)
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def save_csv(df, path, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        old = read_csv_safe(path)
        if not old.empty:
            df = pd.concat([old, df], ignore_index=True)

    df.to_csv(path, index=False)


def find_reference_file():
    for path in REFERENCE_CANDIDATES:
        if path.exists():
            return path

    excluded = {name.lower() for name in REPORTS.values()}
    candidates = []

    for folder in (BASE_DIR, DATA_DIR):
        if folder.exists():
            for path in folder.glob("*.csv"):
                name = path.name.lower()
                if (
                    name not in excluded
                    and not name.startswith(
                        ("modelguard_", "predictions", "simulated_")
                    )
                ):
                    candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    # No local CSV found: use the project's original GitHub dataset URL.
    return REFERENCE_DATA_URL


def reference_source_name(source):
    """Readable source label for either a local file or a URL."""
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        return "IBM Telco Customer Churn (GitHub URL)"
    return Path(source_text).name


# =====================================================
# PSI DRIFT DETECTION
# =====================================================
def numeric_psi(reference, current, bins=10):
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    edges = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return (
            0.0
            if np.isclose(np.mean(ref), np.mean(cur))
            else np.nan
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)

    ref_pct = np.clip(
        ref_counts / max(ref_counts.sum(), 1), 1e-6, None
    )
    cur_pct = np.clip(
        cur_counts / max(cur_counts.sum(), 1), 1e-6, None
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def categorical_psi(reference, current):
    ref = reference.astype("string").fillna("<MISSING>").astype(str)
    cur = current.astype("string").fillna("<MISSING>").astype(str)

    categories = sorted(set(ref.unique()) | set(cur.unique()))

    if not categories:
        return np.nan

    ref_counts = ref.value_counts().reindex(
        categories, fill_value=0
    ).astype(float)

    cur_counts = cur.value_counts().reindex(
        categories, fill_value=0
    ).astype(float)

    ref_pct = np.clip(
        ref_counts / max(ref_counts.sum(), 1), 1e-6, None
    )
    cur_pct = np.clip(
        cur_counts / max(cur_counts.sum(), 1), 1e-6, None
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


# =====================================================
# KS TEST
# =====================================================
def ks_test(reference, current):
    try:
        from scipy.stats import ks_2samp

        ref = pd.to_numeric(
            reference, errors="coerce"
        ).dropna()

        cur = pd.to_numeric(
            current, errors="coerce"
        ).dropna()

        if ref.empty or cur.empty:
            return np.nan, np.nan

        result = ks_2samp(ref, cur)

        return float(result.statistic), float(result.pvalue)

    except Exception:
        return np.nan, np.nan


# =====================================================
# SIMULATE 200–250 ROWS
# =====================================================
def generate_simulated_data(reference_path):
    reference = pd.read_csv(reference_path)

    if reference.empty:
        raise ValueError("The reference dataset is empty.")

    rng = np.random.default_rng()

    row_count = int(rng.integers(200, 251))

    current = reference.sample(
        n=row_count,
        replace=True,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)

    # Add small changes to numeric features.
    # This is simulated data, not real incoming customer data.
    for col in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            original = pd.to_numeric(
                current[col], errors="coerce"
            )

            std = pd.to_numeric(
                reference[col], errors="coerce"
            ).std()

            if pd.notna(std) and std > 0:
                noise = rng.normal(
                    0, float(std) * 0.03, len(current)
                )

                current[col] = original + noise

                if pd.api.types.is_integer_dtype(reference[col]):
                    current[col] = np.round(
                        current[col]
                    ).astype("Int64")

    return current


# =====================================================
# DAILY MONITORING UPDATE
# =====================================================
def run_daily_update(force=False):
    reference_path = find_reference_file()

    if reference_path is None:
        raise FileNotFoundError(
            "Could not find a local reference CSV or load the configured dataset URL. "
            "Check your internet connection and try again."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    marker = DATA_DIR / "last_daily_update.json"

    if marker.exists() and not force:
        try:
            saved = json.loads(
                marker.read_text(encoding="utf-8")
            )

            if saved.get("date") == today:
                return {
                    "status": "Already updated today",
                    "rows": saved.get("rows", 0),
                    "date": today,
                }
        except Exception:
            pass

    current = generate_simulated_data(reference_path)

    timestamp = datetime.now().isoformat(timespec="seconds")

    dated_file = (
        SIMULATED_DIR / f"simulated_current_{today}.csv"
    )

    current.to_csv(dated_file, index=False)
    current.to_csv(DATA_DIR / "simulated_current.csv", index=False)

    reference = pd.read_csv(reference_path)

    psi_rows = []
    ks_rows = []
    quality_rows = []
    alert_rows = []

    for col in reference.columns:
        if col not in current.columns:
            continue

        if pd.api.types.is_numeric_dtype(reference[col]):
            psi = numeric_psi(reference[col], current[col])
            statistic, p_value = ks_test(
                reference[col], current[col]
            )

            ks_rows.append({
                "timestamp": timestamp,
                "feature": col,
                "ks_statistic": statistic,
                "p_value": p_value,
                "significant_at_0.05": (
                    bool(p_value < 0.05)
                    if pd.notna(p_value)
                    else False
                ),
            })
        else:
            psi = categorical_psi(reference[col], current[col])

        severity = (
            "High"
            if pd.notna(psi) and psi >= PSI_HIGH
            else "Review"
            if pd.notna(psi) and psi >= PSI_LOW
            else "Low"
        )

        psi_rows.append({
            "timestamp": timestamp,
            "feature": col,
            "psi": psi,
            "severity": severity,
        })

        if severity in ("High", "Review"):
            alert_rows.append({
                "timestamp": timestamp,
                "feature": col,
                "severity": severity,
                "message": (
                    f"{severity} simulated distribution drift "
                    f"detected for {col}."
                ),
            })

        quality_rows.append({
            "timestamp": timestamp,
            "feature": col,
            "missing_values": int(current[col].isna().sum()),
            "missing_percent": round(
                current[col].isna().mean() * 100, 3
            ),
            "unique_values": int(current[col].nunique()),
            "dtype": str(current[col].dtype),
        })

    psi_df = pd.DataFrame(psi_rows)
    ks_df = pd.DataFrame(ks_rows)
    quality_df = pd.DataFrame(quality_rows)
    alerts_df = pd.DataFrame(alert_rows)

    save_csv(psi_df, BASE_DIR / REPORTS["PSI Report"])
    save_csv(ks_df, BASE_DIR / REPORTS["KS Report"])
    save_csv(quality_df, BASE_DIR / REPORTS["Data Quality"])
    save_csv(
        alerts_df,
        BASE_DIR / REPORTS["Alerts"],
        append=True,
    )

    high_count = int(
        (pd.to_numeric(psi_df["psi"], errors="coerce") >= PSI_HIGH).sum()
    )

    history_row = pd.DataFrame([{
        "timestamp": timestamp,
        "simulated_rows": len(current),
        "reference_file": reference_source_name(reference_path),
        "high_drift_features": high_count,
        "alerts_created": len(alerts_df),
        "simulated_file": str(dated_file.relative_to(BASE_DIR)),
    }])

    save_csv(
        history_row,
        BASE_DIR / REPORTS["Monitoring History"],
        append=True,
    )

    marker.write_text(
        json.dumps({
            "date": today,
            "rows": len(current),
            "dataset": str(dated_file),
            "updated_at": timestamp,
        }, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "Updated",
        "rows": len(current),
        "dataset": str(dated_file),
        "date": today,
        "high_drift_features": high_count,
        "alerts_created": len(alerts_df),
    }


# =====================================================
# DASHBOARD HELPERS
# =====================================================
def load_report(name):
    return read_csv_safe(BASE_DIR / REPORTS[name])


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def show_table(df):
    if df.empty:
        st.info("No saved records available yet.")
    else:
        st.dataframe(
            df, use_container_width=True, hide_index=True
        )


def download_csv(df, filename):
    if not df.empty:
        st.download_button(
            f"Download {filename}",
            df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )


def safe_model_load(path):
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, str(exc)


# =====================================================
# CHART HELPERS
# =====================================================
def show_pie_chart(data, name_col, value_col, title):
    """Display a pie chart using Streamlit's built-in Vega-Lite renderer."""
    chart_data = data[[name_col, value_col]].copy()
    chart_data[name_col] = chart_data[name_col].fillna("Unspecified").astype(str)
    chart_data[value_col] = pd.to_numeric(chart_data[value_col], errors="coerce")
    chart_data = chart_data.dropna()
    chart_data = chart_data[chart_data[value_col] > 0]
    if chart_data.empty:
        st.info("No data available for this pie chart yet.")
        return
    spec = {
        "title": title,
        "data": {"values": chart_data.to_dict(orient="records")},
        "mark": {"type": "arc", "outerRadius": 120, "stroke": "#ffffff", "strokeWidth": 2},
        "encoding": {
            "theta": {"field": value_col, "type": "quantitative"},
            "color": {
                "field": name_col,
                "type": "nominal",
                "title": name_col,
                "scale": {"scheme": "tableau10"},
            },
            "tooltip": [
                {"field": name_col, "type": "nominal", "title": name_col},
                {"field": value_col, "type": "quantitative", "title": value_col},
            ],
        },
        "view": {"stroke": None},
    }
    st.vega_lite_chart(spec, use_container_width=True)


def show_colored_bar_chart(data, category_col, value_col, title, color_scheme="blues"):
    """Render a colorful, tooltip-enabled bar chart with explicit category colors."""
    chart_data = data[[category_col, value_col]].copy()
    chart_data[category_col] = chart_data[category_col].fillna("Unspecified").astype(str)
    chart_data[value_col] = pd.to_numeric(chart_data[value_col], errors="coerce")
    chart_data = chart_data.dropna()
    if chart_data.empty:
        st.info("No data available for this chart yet.")
        return
    spec = {
        "title": title,
        "data": {"values": chart_data.to_dict(orient="records")},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5},
        "encoding": {
            "x": {"field": category_col, "type": "nominal", "sort": "-y",
                  "axis": {"labelAngle": -35, "labelLimit": 180}},
            "y": {"field": value_col, "type": "quantitative"},
            "color": {"field": category_col, "type": "nominal",
                      "scale": {"scheme": color_scheme}, "legend": None},
            "tooltip": [
                {"field": category_col, "type": "nominal", "title": category_col},
                {"field": value_col, "type": "quantitative", "title": value_col},
            ],
        },
        "config": {"view": {"stroke": None}, "axis": {"grid": True}},
    }
    st.vega_lite_chart(spec, use_container_width=True)


# =====================================================
# STREAMLIT DASHBOARD
# =====================================================
def dashboard():
    st.set_page_config(
        page_title="ModelGuard | ML Monitoring",
        page_icon="🛡️",
        layout="wide",
    )

    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.08);
        border: 1px solid rgba(128,128,128,0.25);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ ModelGuard")
    st.caption(
        "Explainable ML Monitoring and Early-Warning Platform"
    )

    with st.sidebar:
        st.header("Monitoring")
        st.write("Daily simulated batch: 200–250 rows")

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Drift Analysis",
                "Data Quality",
                "Alerts & Incidents",
                "Monitoring History",
                "Model Registry",
                "Raw Reports",
            ],
        )

    # Check the date on every Streamlit rerun. The marker prevents duplicate
    # runs on the same date, while a rerun after midnight can create a new batch.
    try:
        result = run_daily_update(force=False)
        if result["status"] == "Updated":
            st.toast(
                f"Daily simulated batch created: {result['rows']} rows"
            )
    except Exception as exc:
        st.warning(f"Daily update unavailable: {exc}")

    marker_path = DATA_DIR / "last_daily_update.json"
    if marker_path.exists():
        try:
            marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
            st.sidebar.caption(f"Last simulated update: {marker_data.get('updated_at', 'Unknown')}")
            st.sidebar.caption(f"Rows in latest batch: {marker_data.get('rows', 'Unknown')}")
        except Exception:
            st.sidebar.caption("Daily update status could not be read.")

    psi = load_report("PSI Report")
    ks = load_report("KS Report")
    quality = load_report("Data Quality")
    alerts = load_report("Alerts")
    history = load_report("Monitoring History")

    if page == "Overview":
        st.header("Dashboard Overview")

        psi_col = find_col(psi, ["psi", "PSI", "psi_value"])
        feature_col = find_col(
            psi, ["feature", "Feature", "column", "Column"]
        )
        severity_col = find_col(
            alerts, ["severity", "Severity", "level", "Level"]
        )

        high_drift = 0
        if psi_col and not psi.empty:
            values = pd.to_numeric(psi[psi_col], errors="coerce")
            high_drift = int((values >= PSI_HIGH).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features Checked", len(psi))
        c2.metric("High PSI Features", high_drift)
        c3.metric("Total Alerts", len(alerts))
        c4.metric("Saved Monitoring Runs", len(history))

        left, right = st.columns(2)

        with left:
            st.subheader("Feature Drift (PSI)")

            if not psi.empty and psi_col and feature_col:
                chart = psi[[feature_col, psi_col]].copy()
                chart[psi_col] = pd.to_numeric(
                    chart[psi_col], errors="coerce"
                )
                chart = chart.dropna()

                if not chart.empty:
                    show_colored_bar_chart(
                        chart, feature_col, psi_col,
                        "PSI by feature (higher values indicate more drift)",
                        color_scheme="redyellowblue",
                    )
                else:
                    st.info("No numeric PSI values available.")
            else:
                st.info("PSI report not available yet.")

        with right:
            st.subheader("Alerts by Severity")

            if not alerts.empty and severity_col:
                counts = (
                    alerts[severity_col]
                    .fillna("Unspecified")
                    .astype(str)
                    .value_counts()
                )
                severity_counts = counts.rename_axis("Severity").reset_index(name="Alerts")
                show_pie_chart(severity_counts, "Severity", "Alerts", "Alerts by severity")
                show_colored_bar_chart(
                    severity_counts, "Severity", "Alerts",
                    "Alert count by severity", color_scheme="tableau10"
                )
            else:
                st.info("No alerts available.")

        # Drift severity distribution as a pie chart.
        if not psi.empty and psi_col:
            psi_values = pd.to_numeric(psi[psi_col], errors="coerce")
            drift_summary = pd.DataFrame({
                "Drift Level": np.select(
                    [psi_values >= PSI_HIGH, psi_values >= PSI_LOW],
                    ["High drift", "Moderate drift"],
                    default="Low drift",
                )
            })
            drift_summary = drift_summary["Drift Level"].value_counts().rename_axis("Drift Level").reset_index(name="Features")
            st.subheader("Drift Level Distribution")
            show_pie_chart(drift_summary, "Drift Level", "Features", "Features by drift level")
            st.caption("PSI guide: below 0.10 = low; 0.10–0.25 = review; 0.25 or above = high drift.")

        st.subheader("Monitoring Pipeline")
        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px;margin:12px 0 22px 0;">
          <div style="border:1px solid #64748b;border-radius:10px;padding:12px;text-align:center;min-width:110px;">Reference<br><b>Dataset</b></div>
          <div style="font-size:24px;">→</div>
          <div style="border:1px solid #64748b;border-radius:10px;padding:12px;text-align:center;min-width:110px;">Daily<br><b>Simulation</b></div>
          <div style="font-size:24px;">→</div>
          <div style="border:1px solid #64748b;border-radius:10px;padding:12px;text-align:center;min-width:110px;">Data<br><b>Quality</b></div>
          <div style="font-size:24px;">→</div>
          <div style="border:1px solid #64748b;border-radius:10px;padding:12px;text-align:center;min-width:110px;">PSI + KS<br><b>Drift Tests</b></div>
          <div style="font-size:24px;">→</div>
          <div style="border:1px solid #64748b;border-radius:10px;padding:12px;text-align:center;min-width:110px;">Alerts &amp;<br><b>Reports</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Latest Monitoring History")
        show_table(history.tail(10))

    elif page == "Drift Analysis":
        st.header("Drift Analysis")
        tab1, tab2 = st.tabs(["PSI Analysis", "KS Test"])

        with tab1:
            show_table(psi)
            pcol = find_col(psi, ["psi", "PSI", "psi_value"])
            fcol = find_col(
                psi, ["feature", "Feature", "column", "Column"]
            )

            if not psi.empty and pcol and fcol:
                chart = psi[[fcol, pcol]].copy()
                chart[pcol] = pd.to_numeric(
                    chart[pcol], errors="coerce"
                )
                chart = chart.dropna()
                if not chart.empty:
                    show_colored_bar_chart(
                        chart, fcol, pcol, "PSI by feature",
                        color_scheme="redyellowblue"
                    )

            download_csv(psi, REPORTS["PSI Report"])

        with tab2:
            show_table(ks)
            scol = find_col(
                ks, ["ks_statistic", "KS Statistic", "statistic"]
            )
            fcol = find_col(
                ks, ["feature", "Feature", "column", "Column"]
            )

            if not ks.empty and scol and fcol:
                chart = ks[[fcol, scol]].copy()
                chart[scol] = pd.to_numeric(
                    chart[scol], errors="coerce"
                )
                chart = chart.dropna()
                if not chart.empty:
                    show_colored_bar_chart(
                        chart, fcol, scol, "KS statistic by feature",
                        color_scheme="viridis"
                    )

            download_csv(ks, REPORTS["KS Report"])

    elif page == "Data Quality":
        st.header("Data Quality")
        show_table(quality)

        missing_col = find_col(
            quality,
            ["missing_values", "Missing Values", "missing_count"],
        )
        feature_col = find_col(
            quality, ["feature", "Feature", "column", "Column"]
        )

        if missing_col and not quality.empty:
            missing = pd.to_numeric(
                quality[missing_col], errors="coerce"
            ).fillna(0)

            st.metric("Total Missing Values", int(missing.sum()))

            if feature_col:
                chart = quality[[feature_col]].copy()
                chart["Missing Values"] = missing
                show_colored_bar_chart(
                    chart, feature_col, "Missing Values",
                    "Missing values by feature", color_scheme="oranges"
                )

        download_csv(quality, REPORTS["Data Quality"])

    elif page == "Alerts & Incidents":
        st.header("Alerts & Incidents")

        if alerts.empty:
            st.success("No saved alerts.")
        else:
            if severity_col := find_col(
                alerts, ["severity", "Severity", "level", "Level"]
            ):
                severity_counts = (
                    alerts[severity_col].fillna("Unspecified").astype(str)
                    .value_counts().rename_axis("Severity").reset_index(name="Alerts")
                )
                show_colored_bar_chart(
                    severity_counts, "Severity", "Alerts",
                    "Alerts by severity", color_scheme="tableau10"
                )

            show_table(alerts.tail(250).iloc[::-1])
            download_csv(alerts, REPORTS["Alerts"])

    elif page == "Monitoring History":
        st.header("Monitoring History")
        show_table(history)

        if not history.empty:
            numeric_cols = history.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_cols:
                metric = st.selectbox(
                    "Choose a metric", numeric_cols
                )
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.subheader("Trend line")
                    st.line_chart(history[metric])
                with chart_col2:
                    st.subheader("Area trend")
                    st.area_chart(history[metric])

        download_csv(history, REPORTS["Monitoring History"])

    elif page == "Model Registry":
        st.header("Model Registry")

        registry_path = BASE_DIR / "models" / "registry.json"

        if registry_path.exists():
            try:
                st.json(json.loads(
                    registry_path.read_text(encoding="utf-8")
                ))
            except Exception as exc:
                st.warning(f"Could not read registry: {exc}")

        model_files = sorted(
            set(
                list(BASE_DIR.glob("*.pkl"))
                + list((BASE_DIR / "models").glob("*.pkl"))
                + list(BASE_DIR.glob("*.keras"))
                + list((BASE_DIR / "models").glob("*.keras"))
            )
        )

        if model_files:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Model File": str(p.relative_to(BASE_DIR)),
                        "Size KB": round(p.stat().st_size / 1024, 2),
                    }
                    for p in model_files
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No model files found.")

        rf_path = BASE_DIR / "random_forest_pipeline.pkl"

        if rf_path.exists() and st.button("Test-load Random Forest"):
            model, error = safe_model_load(rf_path)

            if error:
                st.error(error)
            else:
                st.success(
                    f"Model loaded: {type(model).__name__}"
                )

    elif page == "Raw Reports":
        st.header("Raw Reports")

        for title, filename in REPORTS.items():
            with st.expander(title):
                df = load_report(title)
                show_table(df)
                download_csv(df, filename)

    st.divider()
    st.caption(
        "ModelGuard | Local prototype | Simulated data is not live production data."
    )


# =====================================================
# COMMAND LINE
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--daily-update",
        action="store_true",
        help="Generate today's simulated dataset and reports",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force another update even if today's update exists",
    )

    args, _ = parser.parse_known_args()

    if args.daily_update:
        try:
            result = run_daily_update(force=args.force)
            print(json.dumps(result, indent=2))
        except Exception as exc:
            print(f"Monitoring failed: {exc}", file=sys.stderr)
            raise SystemExit(1)
    else:
        dashboard()


if __name__ == "__main__":
    main()