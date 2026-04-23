import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from fpdf import FPDF
import tempfile
import os


st.set_page_config(page_title="Mines Dashboard", layout="wide")
st.title("Mines Dashboard")

CSV_URL = "https://docs.google.com/spreadsheets/d/1VLGcrqJIE2Exl6rf_iagESGav3FFFP8CuxfVACXrDVQ/gviz/tq?tqx=out:csv&gid=608849772"

df = pd.read_csv(CSV_URL)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Output"] = df["Output"].astype(str).str.replace(",", ".", regex=False)
df["Output"] = pd.to_numeric(df["Output"], errors="coerce")
df["Weekday"] = df["Weekday"].astype(str).str.strip()
df = df.dropna(subset=["Date", "Mine", "Output"]).copy()


def classify_anomaly(output_value, reference_value):
    if pd.isna(reference_value):
        return "Unknown"
    return "Spike" if output_value > reference_value else "Drop"


def safe_pdf_text(text):
    if text is None:
        return ""
    replacements = {
        "\u043f\u043d": "Mon", "\u0432\u0442": "Tue", "\u0441\u0440": "Wed",
        "\u0447\u0442": "Thu", "\u043f\u0442": "Fri", "\u0441\u0431": "Sat",
        "\u0432\u0441": "Sun", "\u041f\u041d": "Mon", "\u0412\u0422": "Tue",
        "\u0421\u0420": "Wed", "\u0427\u0422": "Thu", "\u041f\u0422": "Fri",
        "\u0421\u0411": "Sat", "\u0412\u0421": "Sun",
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u201c": '"', "\u201d": '"'
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def grubbs_flag(series, alpha=0.05):
    s = series.dropna()
    n = len(s)
    if n < 3:
        return pd.Series([False] * len(series), index=series.index)
    mean_val = s.mean()
    std_val = s.std(ddof=1)
    if std_val == 0 or pd.isna(std_val):
        return pd.Series([False] * len(series), index=series.index)
    abs_diff = (s - mean_val).abs()
    max_idx = abs_diff.idxmax()
    G = abs_diff.loc[max_idx] / std_val
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit ** 2) / (n - 2 + t_crit ** 2))
    result = pd.Series([False] * len(series), index=series.index)
    if G > G_crit:
        result.loc[max_idx] = True
    return result


def detect_anomalies(df_input, z_threshold, iqr_multiplier, ma_window, ma_percent, grubbs_alpha):
    result = df_input.copy()
    result["Z_Anomaly"] = False
    result["IQR_Anomaly"] = False
    result["MA_Anomaly"] = False
    result["Grubbs_Anomaly"] = False
    result["z_score"] = np.nan
    result["MA_Distance_Pct"] = np.nan

    for mine in result["Mine"].dropna().unique():
        mask = result["Mine"] == mine
        mine_df = result.loc[mask].sort_values("Date").copy()

        mean_val = mine_df["Output"].mean()
        std_val = mine_df["Output"].std()
        if pd.notna(std_val) and std_val != 0:
            z_scores = (mine_df["Output"] - mean_val) / std_val
            result.loc[mine_df.index, "z_score"] = z_scores
            result.loc[mine_df.index, "Z_Anomaly"] = z_scores.abs() >= z_threshold

        q1 = mine_df["Output"].quantile(0.25)
        q3 = mine_df["Output"].quantile(0.75)
        iqr = q3 - q1
        result.loc[mine_df.index, "IQR_Anomaly"] = (
            (mine_df["Output"] < q1 - iqr_multiplier * iqr) |
            (mine_df["Output"] > q3 + iqr_multiplier * iqr)
        )

        rolling_mean = mine_df["Output"].rolling(window=ma_window, min_periods=2).mean()
        distance_pct = np.where(
            rolling_mean != 0,
            ((mine_df["Output"] - rolling_mean).abs() / rolling_mean.abs()) * 100,
            np.nan
        )
        result.loc[mine_df.index, "MA_Distance_Pct"] = distance_pct
        result.loc[mine_df.index, "MA_Anomaly"] = (
            pd.Series(distance_pct, index=mine_df.index) >= ma_percent
        )

        grubbs_result = grubbs_flag(mine_df["Output"], alpha=grubbs_alpha)
        result.loc[mine_df.index, "Grubbs_Anomaly"] = grubbs_result.values

    result["Any_Anomaly"] = (
        result["Z_Anomaly"] | result["IQR_Anomaly"] |
        result["MA_Anomaly"] | result["Grubbs_Anomaly"]
    )
    return result


def detect_total_anomalies(total_df, z_threshold, iqr_multiplier, ma_window, ma_percent, grubbs_alpha):
    result = total_df.copy().sort_values("Date").reset_index(drop=True)
    result["Z_Anomaly"] = False
    result["IQR_Anomaly"] = False
    result["MA_Anomaly"] = False
    result["Grubbs_Anomaly"] = False
    result["z_score"] = np.nan
    result["MA_Distance_Pct"] = np.nan

    mean_val = result["Output"].mean()
    std_val = result["Output"].std()
    if pd.notna(std_val) and std_val != 0:
        z_scores = (result["Output"] - mean_val) / std_val
        result["z_score"] = z_scores
        result["Z_Anomaly"] = z_scores.abs() >= z_threshold

    q1 = result["Output"].quantile(0.25)
    q3 = result["Output"].quantile(0.75)
    iqr = q3 - q1
    result["IQR_Anomaly"] = (
        (result["Output"] < q1 - iqr_multiplier * iqr) |
        (result["Output"] > q3 + iqr_multiplier * iqr)
    )

    rolling_mean = result["Output"].rolling(window=ma_window, min_periods=2).mean()
    distance_pct = np.where(
        rolling_mean != 0,
        ((result["Output"] - rolling_mean).abs() / rolling_mean.abs()) * 100,
        np.nan
    )
    result["MA_Distance_Pct"] = distance_pct
    result["MA_Anomaly"] = pd.Series(distance_pct) >= ma_percent

    grubbs_result = grubbs_flag(result["Output"], alpha=grubbs_alpha)
    result["Grubbs_Anomaly"] = grubbs_result.values

    result["Any_Anomaly"] = (
        result["Z_Anomaly"] | result["IQR_Anomaly"] |
        result["MA_Anomaly"] | result["Grubbs_Anomaly"]
    )
    return result


def add_outlier_and_trend_traces(fig, chart_df, trend_degree):
    for mine in chart_df["Mine"].unique():
        mine_data = chart_df[chart_df["Mine"] == mine].sort_values("Date")
        anomaly_pts = mine_data[mine_data["Any_Anomaly"]]
        if not anomaly_pts.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_pts["Date"],
                y=anomaly_pts["Output"],
                mode="markers",
                name=f"{mine} anomalies",
                marker=dict(size=12, symbol="x", color="red", line=dict(width=2))
            ))
        if mine_data["Trend"].notna().any():
            fig.add_trace(go.Scatter(
                x=mine_data["Date"],
                y=mine_data["Trend"],
                mode="lines",
                name=f"{mine} trend",
                line=dict(dash="dash", width=2)
            ))
    return fig


def save_plotly_figure(fig, filename_prefix="chart"):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=filename_prefix)
    fig.write_image(tmp_file.name, width=1400, height=700, scale=2)
    return tmp_file.name


class PDFReport(FPDF):
    C_DARK   = (30, 30, 40)
    C_ACCENT = (52, 100, 180)
    C_SPIKE  = (200, 60, 60)
    C_DROP   = (220, 130, 30)
    C_LIGHT  = (245, 246, 250)
    C_WHITE  = (255, 255, 255)
    C_MUTED  = (110, 115, 130)

    def header(self):
        self.set_fill_color(*self.C_DARK)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(4)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*self.C_WHITE)
        self.cell(0, 10, "WEYLAND-YUTANI CORPORATION  |  Mining Operations Report", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*self.C_DARK)
        self.rect(0, 285, 210, 12, "F")
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*self.C_WHITE)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, title, number=None):
        self.ln(3)
        self.set_fill_color(*self.C_ACCENT)
        self.rect(self.l_margin, self.get_y(), 190, 8, "F")
        self.set_font("helvetica", "B", 10)
        self.set_text_color(*self.C_WHITE)
        label = f"  {number}.  {title}" if number else f"  {title}"
        self.cell(190, 8, safe_pdf_text(label), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, key, value, shade=False):
        if shade:
            self.set_fill_color(*self.C_LIGHT)
            self.rect(self.l_margin, self.get_y(), 190, 7, "F")
        self.set_font("helvetica", "B", 9)
        self.set_text_color(*self.C_MUTED)
        self.cell(70, 7, safe_pdf_text(str(key)))
        self.set_font("helvetica", "", 9)
        self.set_text_color(*self.C_DARK)
        self.multi_cell(120, 7, safe_pdf_text(str(value)))
        self.set_text_color(0, 0, 0)

    def metric_box(self, label, value, x, y, w=44, h=18):
        self.set_fill_color(*self.C_LIGHT)
        self.rect(x, y, w, h, "F")
        self.set_draw_color(*self.C_ACCENT)
        self.rect(x, y, w, h)
        self.set_draw_color(0, 0, 0)
        self.set_xy(x, y + 2)
        self.set_font("helvetica", "", 7)
        self.set_text_color(*self.C_MUTED)
        self.cell(w, 5, safe_pdf_text(label.upper()), align="C")
        self.set_xy(x, y + 7)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*self.C_DARK)
        self.cell(w, 8, safe_pdf_text(str(value)), align="C")
        self.set_text_color(0, 0, 0)

    def stats_table(self, dataframe):
        if dataframe.empty:
            self.set_font("helvetica", "I", 9)
            self.cell(0, 7, "No data available.", ln=True)
            return
        cols = list(dataframe.columns)
        col_w = 190 / len(cols)
        self.set_fill_color(*self.C_DARK)
        self.set_font("helvetica", "B", 8)
        self.set_text_color(*self.C_WHITE)
        for col in cols:
            self.cell(col_w, 7, safe_pdf_text(str(col))[:18], border=0, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        for i, (_, row) in enumerate(dataframe.iterrows()):
            self.set_fill_color(*self.C_LIGHT if i % 2 == 0 else self.C_WHITE)
            self.set_font("helvetica", "", 8)
            for val in row:
                self.cell(col_w, 6, safe_pdf_text(str(val))[:18], border=0, fill=True, align="C")
            self.ln()
        self.ln(3)

    def anomaly_card(self, date_str, mine_name, output_val, atype,
                     mean_ref, tests, z_val, ma_val):
        color = self.C_SPIKE if atype == "Spike" else self.C_DROP
        card_y = self.get_y()
        if card_y > 260:
            self.add_page()
            card_y = self.get_y()
        self.set_fill_color(*self.C_LIGHT)
        self.rect(self.l_margin, card_y, 190, 22, "F")
        self.set_fill_color(*color)
        self.rect(self.l_margin, card_y, 4, 22, "F")
        self.set_xy(self.l_margin + 6, card_y + 2)
        self.set_font("helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(30, 5, safe_pdf_text(f"[{atype.upper()}]"))
        self.set_text_color(*self.C_DARK)
        self.cell(0, 5, safe_pdf_text(f"{date_str}  |  {mine_name}"), ln=True)
        self.set_x(self.l_margin + 6)
        self.set_font("helvetica", "", 8)
        self.set_text_color(*self.C_MUTED)
        self.cell(62, 4, safe_pdf_text(f"Output: {output_val:.2f}"))
        self.cell(62, 4, safe_pdf_text(f"Mine mean: {mean_ref:.2f}"))
        self.cell(0, 4, safe_pdf_text(f"Tests: {tests}"), ln=True)
        self.set_x(self.l_margin + 6)
        self.cell(62, 4, safe_pdf_text(f"Z-score: {z_val}"))
        self.cell(0, 4, safe_pdf_text(f"MA distance: {ma_val}"), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_image_from_path(self, image_path, w=185):
        self.image(image_path, w=w)
        self.ln(4)


def build_pdf_report(
    analyzed_df, filtered_df, mine_stats, total_stats, display_anomalies,
    total_by_day, fig_mines, fig_total, selected_mines, selected_weekdays,
    date_range, chart_type, trend_degree, z_threshold, iqr_multiplier,
    ma_window, ma_percent, grubbs_alpha
):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.section_title("Report Summary", number=1)
    items = [
        ("Selected mines", ", ".join(map(str, selected_mines)) if selected_mines else "None"),
        ("Selected weekdays", ", ".join(map(str, selected_weekdays)) if selected_weekdays else "None"),
        ("Date range", f"{date_range[0]} to {date_range[1]}" if len(date_range) == 2 else "-"),
        ("Chart type", chart_type),
        ("Trendline degree", f"Polynomial degree {trend_degree}"),
    ]
    for i, (k, v) in enumerate(items):
        pdf.kv_row(k, v, shade=(i % 2 == 0))

    pdf.ln(4)
    pdf.section_title("Anomaly Detection Settings", number=2)
    settings = [
        ("Z-score threshold", z_threshold),
        ("IQR multiplier", iqr_multiplier),
        ("Moving average window", ma_window),
        ("Distance from MA (%)", ma_percent),
        ("Grubbs alpha", grubbs_alpha),
    ]
    for i, (k, v) in enumerate(settings):
        pdf.kv_row(k, v, shade=(i % 2 == 0))

    pdf.ln(4)
    pdf.section_title("Key Metrics", number=3)
    pdf.ln(2)
    metrics = [
        ("Total output", f"{filtered_df['Output'].sum():,.2f}"),
        ("Avg daily output", f"{filtered_df['Output'].mean():,.2f}"),
        ("Mines selected", str(filtered_df["Mine"].nunique())),
        ("Days in selection", str(filtered_df["Date"].nunique())),
    ]
    x_start = pdf.l_margin
    y_pos = pdf.get_y()
    for i, (lbl, val) in enumerate(metrics):
        pdf.metric_box(lbl, val, x=x_start + i * 48, y=y_pos)
    pdf.ln(22)

    pdf.add_page()
    pdf.section_title("Charts", number=4)
    mines_chart_path = save_plotly_figure(fig_mines, "mines_chart_")
    total_chart_path = save_plotly_figure(fig_total, "total_chart_")

    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*PDFReport.C_MUTED)
    pdf.cell(0, 6, "Mine production over time", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.add_image_from_path(mines_chart_path)

    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*PDFReport.C_MUTED)
    pdf.cell(0, 6, "Total production over time", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.add_image_from_path(total_chart_path)

    pdf.add_page()
    pdf.section_title("Mine Statistics", number=5)
    pdf.stats_table(mine_stats.round(2))
    pdf.section_title("Total Output Statistics", number=6)
    pdf.stats_table(total_stats.round(2))

    pdf.add_page()
    pdf.section_title("Detected Anomalies (first 30)", number=7)
    if display_anomalies.empty:
        pdf.set_font("helvetica", "I", 9)
        pdf.cell(0, 7, "No anomalies detected for the selected filters.", ln=True)
    else:
        pdf.stats_table(display_anomalies.head(30))

    pdf.add_page()
    pdf.section_title("Anomaly Details by Mine", number=8)
    anomalies_full = analyzed_df[analyzed_df["Any_Anomaly"]].copy()

    if anomalies_full.empty:
        pdf.set_font("helvetica", "I", 9)
        pdf.cell(0, 7, "No anomalies detected.", ln=True)
    else:
        for mine in sorted(anomalies_full["Mine"].dropna().unique()):
            mine_anoms = anomalies_full[anomalies_full["Mine"] == mine].sort_values("Date")
            mine_base_mean = analyzed_df[analyzed_df["Mine"] == mine]["Output"].mean()
            pdf.set_font("helvetica", "B", 9)
            pdf.set_text_color(*PDFReport.C_ACCENT)
            pdf.cell(0, 7, safe_pdf_text(f"Mine: {mine}  ({len(mine_anoms)} anomalies)"), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
            for _, row in mine_anoms.iterrows():
                atype = classify_anomaly(row["Output"], mine_base_mean)
                tests = []
                if row["Z_Anomaly"]: tests.append("Z-score")
                if row["IQR_Anomaly"]: tests.append("IQR")
                if row["MA_Anomaly"]: tests.append("MA dist.")
                if row["Grubbs_Anomaly"]: tests.append("Grubbs")
                z_val = f"{row['z_score']:.2f}" if pd.notna(row["z_score"]) else "N/A"
                ma_val = f"{row['MA_Distance_Pct']:.1f}%" if pd.notna(row["MA_Distance_Pct"]) else "N/A"
                pdf.anomaly_card(
                    date_str=row["Date"].strftime("%Y-%m-%d"),
                    mine_name=mine,
                    output_val=row["Output"],
                    atype=atype,
                    mean_ref=mine_base_mean,
                    tests=", ".join(tests) if tests else "None",
                    z_val=z_val,
                    ma_val=ma_val
                )
            pdf.ln(2)

    pdf.add_page()
    pdf.section_title("Total Output Anomaly Details", number=9)
    total_reference = total_by_day["Output"].mean()
    total_anomalies = total_by_day[total_by_day["Any_Anomaly"]].copy()

    if total_anomalies.empty:
        pdf.set_font("helvetica", "I", 9)
        pdf.cell(0, 7, "No total output anomalies detected.", ln=True)
    else:
        for _, row in total_anomalies.iterrows():
            atype = classify_anomaly(row["Output"], total_reference)
            trend_val = f"{row['Trend']:.2f}" if pd.notna(row.get("Trend")) else "N/A"
            color = PDFReport.C_SPIKE if atype == "Spike" else PDFReport.C_DROP
            card_y = pdf.get_y()
            if card_y > 262:
                pdf.add_page()
                card_y = pdf.get_y()
            pdf.set_fill_color(*PDFReport.C_LIGHT)
            pdf.rect(pdf.l_margin, card_y, 190, 18, "F")
            pdf.set_fill_color(*color)
            pdf.rect(pdf.l_margin, card_y, 4, 18, "F")
            pdf.set_xy(pdf.l_margin + 6, card_y + 2)
            pdf.set_font("helvetica", "B", 9)
            pdf.set_text_color(*color)
            pdf.cell(30, 5, safe_pdf_text(f"[{atype.upper()}]"))
            pdf.set_text_color(*PDFReport.C_DARK)
            pdf.cell(0, 5, safe_pdf_text(row["Date"].strftime("%Y-%m-%d")), ln=True)
            pdf.set_x(pdf.l_margin + 6)
            pdf.set_font("helvetica", "", 8)
            pdf.set_text_color(*PDFReport.C_MUTED)
            pdf.cell(65, 4, safe_pdf_text(f"Total output: {row['Output']:.2f}"))
            pdf.cell(65, 4, safe_pdf_text(f"Mean reference: {total_reference:.2f}"))
            pdf.cell(0, 4, safe_pdf_text(f"Trend value: {trend_val}"), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

    pdf.add_page()
    pdf.section_title("Raw Filtered Data (first 40 rows)", number=10)
    raw_pdf = analyzed_df.sort_values(["Date", "Mine"]).copy()
    raw_pdf["Date"] = raw_pdf["Date"].dt.strftime("%Y-%m-%d")
    raw_pdf["z_score"] = raw_pdf["z_score"].round(2)
    raw_pdf["MA_Distance_Pct"] = raw_pdf["MA_Distance_Pct"].round(2)
    raw_pdf = raw_pdf.rename(columns={
        "Z_Anomaly": "Z anom", "IQR_Anomaly": "IQR anom",
        "MA_Anomaly": "MA anom", "Grubbs_Anomaly": "Grubbs",
        "z_score": "Z-score", "MA_Distance_Pct": "MA dist%",
        "Any_Anomaly": "Any"
    })
    pdf.stats_table(raw_pdf.head(40))

    pdf_bytes = bytes(pdf.output())
    for path in [mines_chart_path, total_chart_path]:
        if os.path.exists(path):
            os.remove(path)
    return pdf_bytes


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
min_date = df["Date"].min()
max_date = df["Date"].max()

selected_mines = st.sidebar.multiselect(
    "Select mines",
    options=sorted(df["Mine"].dropna().unique()),
    default=sorted(df["Mine"].dropna().unique())
)
selected_weekdays = st.sidebar.multiselect(
    "Select weekdays",
    options=sorted(df["Weekday"].dropna().unique()),
    default=sorted(df["Weekday"].dropna().unique())
)
date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

st.sidebar.subheader("Chart settings")
chart_type = st.sidebar.selectbox("Chart type", ["Line", "Bar", "Stacked"])
trend_degree = st.sidebar.selectbox("Trendline degree", [1, 2, 3, 4])

st.sidebar.subheader("Anomaly settings")
z_threshold    = st.sidebar.number_input("Z-score threshold", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
iqr_multiplier = st.sidebar.number_input("IQR multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
ma_window      = st.sidebar.number_input("Moving average window", min_value=2, max_value=30, value=5, step=1)
ma_percent     = st.sidebar.number_input("Distance from moving average (%)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
grubbs_alpha   = st.sidebar.selectbox("Grubbs alpha", options=[0.10, 0.05, 0.01], index=1)

# ── Filter ───────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Mine"].isin(selected_mines)) &
        (df["Weekday"].isin(selected_weekdays)) &
        (df["Date"].dt.date >= start_date) &
        (df["Date"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = df[
        (df["Mine"].isin(selected_mines)) &
        (df["Weekday"].isin(selected_weekdays))
    ].copy()

st.success("Data loaded successfully")

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ── Detect anomalies ─────────────────────────────────────────────────────────
analyzed_df = detect_anomalies(
    filtered_df,
    z_threshold=z_threshold, iqr_multiplier=iqr_multiplier,
    ma_window=ma_window, ma_percent=ma_percent, grubbs_alpha=grubbs_alpha
)

chart_df = analyzed_df.copy().sort_values(["Mine", "Date"])
chart_df["Trend"] = np.nan

for mine in chart_df["Mine"].dropna().unique():
    mine_mask = chart_df["Mine"] == mine
    mine_part = chart_df.loc[mine_mask].sort_values("Date").copy()
    if len(mine_part) > trend_degree:
        x = np.arange(len(mine_part))
        coeffs = np.polyfit(x, mine_part["Output"].values, trend_degree)
        chart_df.loc[mine_part.index, "Trend"] = np.polyval(coeffs, x)

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("Key metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total output",    f"{filtered_df['Output'].sum():,.2f}")
col2.metric("Average output",  f"{filtered_df['Output'].mean():,.2f}")
col3.metric("Mines selected",  filtered_df["Mine"].nunique())
col4.metric("Days in selection", filtered_df["Date"].nunique())

# ── Mine production chart ────────────────────────────────────────────────────
st.subheader("Mine production over time")
fig_mines = go.Figure()

if chart_type == "Line":
    for mine in chart_df["Mine"].unique():
        mine_data = chart_df[chart_df["Mine"] == mine].sort_values("Date")
        fig_mines.add_trace(go.Scatter(
            x=mine_data["Date"], y=mine_data["Output"],
            mode="lines+markers", name=mine
        ))
        anom_pts = mine_data[mine_data["Any_Anomaly"]]
        if not anom_pts.empty:
            fig_mines.add_trace(go.Scatter(
                x=anom_pts["Date"], y=anom_pts["Output"],
                mode="markers", name=f"{mine} anomalies",
                marker=dict(size=12, symbol="x", color="red", line=dict(width=2))
            ))
        if mine_data["Trend"].notna().any():
            fig_mines.add_trace(go.Scatter(
                x=mine_data["Date"], y=mine_data["Trend"],
                mode="lines", name=f"{mine} trend",
                line=dict(dash="dash", width=2)
            ))

elif chart_type == "Bar":
    grouped = chart_df.groupby(["Date", "Mine"], as_index=False)["Output"].sum()
    fig_mines = px.bar(grouped, x="Date", y="Output", color="Mine", barmode="group")
    fig_mines = add_outlier_and_trend_traces(fig_mines, chart_df, trend_degree)

elif chart_type == "Stacked":
    grouped = chart_df.groupby(["Date", "Mine"], as_index=False)["Output"].sum()
    fig_mines = px.bar(grouped, x="Date", y="Output", color="Mine", barmode="stack")
    fig_mines = add_outlier_and_trend_traces(fig_mines, chart_df, trend_degree)

fig_mines.update_layout(
    template="plotly_dark", xaxis_title="Date", yaxis_title="Output", height=500
)
st.plotly_chart(fig_mines, use_container_width=True)

# ── Total production chart ───────────────────────────────────────────────────
st.subheader("Total production over time")

total_by_day = (
    analyzed_df.groupby("Date", as_index=False)
    .agg(Output=("Output", "sum"))
    .sort_values("Date")
)
total_by_day = detect_total_anomalies(
    total_by_day, z_threshold=z_threshold, iqr_multiplier=iqr_multiplier,
    ma_window=ma_window, ma_percent=ma_percent, grubbs_alpha=grubbs_alpha
)
total_by_day = total_by_day.reset_index(drop=True)

if len(total_by_day) > trend_degree:
    x_total = np.arange(len(total_by_day))
    coeffs_total = np.polyfit(x_total, total_by_day["Output"].values, trend_degree)
    total_by_day["Trend"] = np.polyval(coeffs_total, x_total)
else:
    total_by_day["Trend"] = np.nan

fig_total = go.Figure()
fig_total.add_trace(go.Scatter(
    x=total_by_day["Date"], y=total_by_day["Output"],
    mode="lines+markers", name="Total output"
))
total_anom_pts = total_by_day[total_by_day["Any_Anomaly"]]
if not total_anom_pts.empty:
    fig_total.add_trace(go.Scatter(
        x=total_anom_pts["Date"], y=total_anom_pts["Output"],
        mode="markers", name="Total anomalies",
        marker=dict(size=12, symbol="x", color="red", line=dict(width=2))
    ))
if total_by_day["Trend"].notna().any():
    fig_total.add_trace(go.Scatter(
        x=total_by_day["Date"], y=total_by_day["Trend"],
        mode="lines", name="Total trend", line=dict(dash="dash", width=2)
    ))

fig_total.update_layout(
    template="plotly_dark", xaxis_title="Date", yaxis_title="Total output", height=500
)
st.plotly_chart(fig_total, use_container_width=True)

# ── Output by mine ────────────────────────────────────────────────────────────
st.subheader("Total output by mine")
output_by_mine = (
    filtered_df.groupby("Mine", as_index=False)["Output"].sum()
    .sort_values("Output", ascending=False).set_index("Mine")
)
st.bar_chart(output_by_mine, use_container_width=True)

# ── Statistics ────────────────────────────────────────────────────────────────
st.subheader("Mine statistics")
mine_stats = filtered_df.groupby("Mine")["Output"].agg(
    Mean="mean", Std_Dev="std", Median="median"
).reset_index()
q1 = filtered_df.groupby("Mine")["Output"].quantile(0.25)
q3 = filtered_df.groupby("Mine")["Output"].quantile(0.75)
mine_stats["IQR"] = mine_stats["Mine"].map((q3 - q1).to_dict())
mine_stats = mine_stats.round(2)
st.dataframe(mine_stats, use_container_width=True)

total_stats = pd.DataFrame({
    "Mine": ["TOTAL"],
    "Mean": [total_by_day["Output"].mean()],
    "Std_Dev": [total_by_day["Output"].std()],
    "Median": [total_by_day["Output"].median()],
    "IQR": [total_by_day["Output"].quantile(0.75) - total_by_day["Output"].quantile(0.25)]
}).round(2)

st.subheader("Total output statistics")
st.dataframe(total_stats, use_container_width=True)

# ── Anomaly table ─────────────────────────────────────────────────────────────
st.subheader("Detected anomalies")
anomalies = analyzed_df[analyzed_df["Any_Anomaly"]].copy()

if anomalies.empty:
    display_anomalies = pd.DataFrame(columns=[
        "Date", "Mine", "Output", "Z-score", "Z-score anomaly",
        "IQR anomaly", "Distance from MA (%)", "MA anomaly", "Grubbs anomaly"
    ])
    st.info("No anomalies detected for the selected filters.")
else:
    display_anomalies = anomalies[[
        "Date", "Mine", "Output", "z_score", "Z_Anomaly",
        "IQR_Anomaly", "MA_Distance_Pct", "MA_Anomaly", "Grubbs_Anomaly"
    ]].copy()
    display_anomalies["Date"] = display_anomalies["Date"].dt.strftime("%Y-%m-%d")
    display_anomalies["z_score"] = display_anomalies["z_score"].round(2)
    display_anomalies["MA_Distance_Pct"] = display_anomalies["MA_Distance_Pct"].round(2)
    display_anomalies = display_anomalies.rename(columns={
        "z_score": "Z-score", "Z_Anomaly": "Z-score anomaly",
        "IQR_Anomaly": "IQR anomaly", "MA_Distance_Pct": "Distance from MA (%)",
        "MA_Anomaly": "MA anomaly", "Grubbs_Anomaly": "Grubbs anomaly"
    })
    st.dataframe(display_anomalies.sort_values(["Date", "Mine"]), use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
st.subheader("Raw filtered data")
raw_view = analyzed_df.sort_values(["Date", "Mine"]).copy()
raw_view["Date"] = raw_view["Date"].dt.strftime("%Y-%m-%d")
raw_view["z_score"] = raw_view["z_score"].round(2)
raw_view["MA_Distance_Pct"] = raw_view["MA_Distance_Pct"].round(2)
st.dataframe(raw_view, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export report")

if st.button("Generate PDF Report"):
    with st.spinner("Building PDF..."):
        pdf_bytes = build_pdf_report(
            analyzed_df=analyzed_df,
            filtered_df=filtered_df,
            mine_stats=mine_stats,
            total_stats=total_stats,
            display_anomalies=display_anomalies,
            total_by_day=total_by_day,
            fig_mines=fig_mines,
            fig_total=fig_total,
            selected_mines=selected_mines,
            selected_weekdays=selected_weekdays,
            date_range=date_range,
            chart_type=chart_type,
            trend_degree=trend_degree,
            z_threshold=z_threshold,
            iqr_multiplier=iqr_multiplier,
            ma_window=ma_window,
            ma_percent=ma_percent,
            grubbs_alpha=grubbs_alpha
        )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="mines_dashboard_report.pdf",
        mime="application/pdf"
    )
