"""
PEP RALLY Tracker
=================
Local Streamlit dashboard for peptide protocol tracking and fitness optimization.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

APP_TITLE = "PEP RALLY Tracker"
DB_PATH = Path(__file__).resolve().parent / "data" / "pep_rally.db"

# Baseline profile (edit if your starting point changes)
START_WEIGHT_LBS = 213.0
START_DATE = date(2026, 4, 20)  # late April 2026
GOAL_WEIGHT_LBS = 190.0

# Protocol weight history (seeded once; never overwrites existing log dates)
# Format: (YYYY-MM-DD, weight_lbs, note)
HISTORICAL_WEIGHTS: list[tuple[str, float, str]] = [
    ("2026-04-20", 213.0, "Start weight"),
    ("2026-05-20", 207.0, "Check-in"),
    ("2026-05-22", 204.4, "Check-in"),
    ("2026-05-26", 205.8, "Check-in"),
    ("2026-06-08", 210.4, "Post-travel high"),
    ("2026-06-09", 204.0, "Check-in"),
    ("2026-07-01", 200.0, "Check-in"),
    ("2026-07-06", 202.0, "Check-in"),
    ("2026-07-18", 201.5, "Current weight as of July 18, 2026"),
]

# Current protocol stack (display only; dose log captures actuals)
CURRENT_STACK = [
    {
        "peptide": "Retatrutide",
        "dose": "45 units",
        "schedule": "Every Sunday morning",
        "status": "Active",
    },
    {
        "peptide": "KLOW",
        "dose": "13 units",
        "schedule": "Nightly",
        "status": "Active",
    },
    {
        "peptide": "Tesamorelin",
        "dose": "9–10 units (1 mg)",
        "schedule": "Nightly",
        "status": "Active",
    },
    {
        "peptide": "Sermorelin",
        "dose": "12 units",
        "schedule": "Nightly",
        "status": "Active",
    },
    {
        "peptide": "CJC / Ipamorelin",
        "dose": "—",
        "schedule": "Discontinued",
        "status": "Off",
    },
]

PEPTIDE_NAMES = ["Retatrutide", "KLOW", "Tesamorelin", "Sermorelin"]

# Known reconstitutions for the calculator presets
RECON_PRESETS = {
    "Custom": {"vial_mg": 10.0, "bac_ml": 2.0},
    "Reta 24 mg + 2 mL": {"vial_mg": 24.0, "bac_ml": 2.0},
    "Tesa 22 mg + 2 mL": {"vial_mg": 22.0, "bac_ml": 2.0},
}

# Plotly dark layout shared across charts.
# Axis styling is kept separate so callers can pass yaxis=/xaxis= without
# colliding with **PLOTLY_LAYOUT (which previously caused TypeError on yaxis).
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E8EAF0", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
)
AXIS_STYLE = dict(gridcolor="#2A3142", zerolinecolor="#2A3142")


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory and durable write settings."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Durable local writes so form saves survive refresh / process restart
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they do not exist and seed baseline weight anchors."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date    TEXT    NOT NULL UNIQUE,
                weight_lbs  REAL,
                energy      INTEGER,
                sleep       INTEGER,
                hunger      INTEGER,
                joint_pain  INTEGER,
                alcohol     INTEGER NOT NULL DEFAULT 0,
                alcohol_amt TEXT,
                notes       TEXT,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dose_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date    TEXT    NOT NULL,
                peptide     TEXT    NOT NULL,
                units       REAL    NOT NULL,
                notes       TEXT,
                created_at  TEXT    NOT NULL,
                UNIQUE(log_date, peptide)
            );
            """
        )
        # Seed known historical weight check-ins without overwriting any
        # date the user (or a prior seed) already logged.
        conn.executemany(
            """
            INSERT INTO daily_logs (
                log_date, weight_lbs, energy, sleep, hunger, joint_pain,
                alcohol, alcohol_amt, notes, created_at
            ) VALUES (?, ?, NULL, NULL, NULL, NULL, 0, NULL, ?, ?)
            ON CONFLICT(log_date) DO NOTHING
            """,
            [
                (d, w, note, f"{d}T00:00:00")
                for d, w, note in HISTORICAL_WEIGHTS
            ],
        )
        conn.commit()
    finally:
        conn.close()


def upsert_daily_log(row: dict[str, Any]) -> None:
    """Insert or replace a daily log entry by date. Commits immediately."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO daily_logs (
                log_date, weight_lbs, energy, sleep, hunger, joint_pain,
                alcohol, alcohol_amt, notes, created_at
            ) VALUES (
                :log_date, :weight_lbs, :energy, :sleep, :hunger, :joint_pain,
                :alcohol, :alcohol_amt, :notes, :created_at
            )
            ON CONFLICT(log_date) DO UPDATE SET
                weight_lbs  = excluded.weight_lbs,
                energy      = excluded.energy,
                sleep       = excluded.sleep,
                hunger      = excluded.hunger,
                joint_pain  = excluded.joint_pain,
                alcohol     = excluded.alcohol,
                alcohol_amt = excluded.alcohol_amt,
                notes       = excluded.notes,
                created_at  = excluded.created_at
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def upsert_dose_log(row: dict[str, Any]) -> None:
    """Insert or replace a dose entry for a peptide on a given date. Commits immediately."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO dose_logs (log_date, peptide, units, notes, created_at)
            VALUES (:log_date, :peptide, :units, :notes, :created_at)
            ON CONFLICT(log_date, peptide) DO UPDATE SET
                units      = excluded.units,
                notes      = excluded.notes,
                created_at = excluded.created_at
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def load_daily_logs() -> pd.DataFrame:
    """Load all daily logs ordered by date ascending."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM daily_logs ORDER BY log_date ASC",
            conn,
        )
    finally:
        conn.close()
    if not df.empty:
        df["log_date"] = pd.to_datetime(df["log_date"]).dt.date
    return df


def load_dose_logs(days: int | None = None) -> pd.DataFrame:
    """Load dose logs, optionally limited to the last N calendar days."""
    conn = get_connection()
    try:
        if days is not None:
            cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
            df = pd.read_sql_query(
                """
                SELECT * FROM dose_logs
                WHERE log_date >= ?
                ORDER BY log_date DESC, peptide ASC
                """,
                conn,
                params=(cutoff,),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM dose_logs ORDER BY log_date DESC, peptide ASC",
                conn,
            )
    finally:
        conn.close()
    if not df.empty:
        df["log_date"] = pd.to_datetime(df["log_date"]).dt.date
    return df


def get_latest_weight() -> float | None:
    """Return the most recent logged weight, if any."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT weight_lbs FROM daily_logs
            WHERE weight_lbs IS NOT NULL
            ORDER BY log_date DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    return float(row["weight_lbs"]) if row else None


def export_table_csv(table: str) -> bytes:
    """Export a full table as CSV bytes for download."""
    conn = get_connection()
    try:
        # Whitelist table names — never interpolate untrusted input into SQL
        if table not in ("daily_logs", "dose_logs"):
            raise ValueError(f"Unknown table: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY 1", conn)
    finally:
        conn.close()
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


def days_on_protocol() -> int:
    """Days elapsed since START_DATE (inclusive of today)."""
    return max((date.today() - START_DATE).days + 1, 0)


def compute_weight_stats(df: pd.DataFrame) -> dict[str, float | None]:
    """
    Derive current weight, total lost, and average weekly loss.

    Total lost is always Start → latest logged weight.
    Avg weekly loss uses the first and last dated weight entries in the
    chart series so the full April→present trend is reflected.
    """
    current = get_latest_weight()
    total_lost = (START_WEIGHT_LBS - current) if current is not None else None

    avg_weekly: float | None = None
    weight_df = df.dropna(subset=["weight_lbs"]).copy() if not df.empty else df
    if not weight_df.empty and len(weight_df) >= 2:
        weight_df = weight_df.sort_values("log_date")
        first = weight_df.iloc[0]
        last = weight_df.iloc[-1]
        day_span = (last["log_date"] - first["log_date"]).days
        if day_span > 0:
            # Prefer protocol start weight when the series begins at START_DATE
            start_w = float(first["weight_lbs"])
            if first["log_date"] == START_DATE:
                start_w = START_WEIGHT_LBS
            lost = start_w - float(last["weight_lbs"])
            avg_weekly = lost / (day_span / 7.0)

    return {
        "current": current,
        "total_lost": total_lost,
        "avg_weekly": avg_weekly,
    }


def calc_reconstitution(
    vial_mg: float,
    bac_ml: float,
    desired_dose: float,
    dose_unit: str,
) -> dict[str, float]:
    """
    Peptide reconstitution math for insulin syringes (U-100).

    1 mL = 100 units on a standard U-100 insulin syringe.
    Concentration is mg/mL; doses may be entered as mcg or mg.
    """
    if vial_mg <= 0 or bac_ml <= 0:
        raise ValueError("Vial size and BAC volume must be positive.")

    concentration_mg_ml = vial_mg / bac_ml  # mg per mL
    dose_mg = desired_dose / 1000.0 if dose_unit == "mcg" else desired_dose

    if dose_mg <= 0:
        raise ValueError("Desired dose must be positive.")

    volume_ml = dose_mg / concentration_mg_ml
    units = volume_ml * 100.0  # U-100 syringe

    return {
        "concentration_mg_ml": concentration_mg_ml,
        "dose_mg": dose_mg,
        "volume_ml": volume_ml,
        "units": units,
        "mcg_per_unit": (concentration_mg_ml * 1000.0) / 100.0,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    """Custom CSS for a clean, high-contrast dark dashboard."""
    st.markdown(
        """
        <style>
        /* Tighten default padding slightly for denser mobile use */
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #1A1F2E;
            border: 1px solid #2A3142;
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetric"] label {
            color: #9AA3B5 !important;
            font-size: 0.85rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FAFAFA !important;
            font-weight: 650;
        }

        /* Section headers */
        h2, h3 {
            letter-spacing: 0.01em;
            margin-top: 0.4rem !important;
        }

        /* Subtle divider spacing */
        hr {
            margin: 1.1rem 0 1.25rem 0;
            border-color: #2A3142;
        }

        /* Stack status pills via markdown badges */
        .stack-active {
            color: #3DDC97;
            font-weight: 600;
        }
        .stack-off {
            color: #FF6B6B;
            font-weight: 600;
        }

        /* Form submit / primary buttons already themed via config */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }

        /* Dataframe polish */
        div[data-testid="stDataFrame"] {
            border: 1px solid #2A3142;
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(stats: dict[str, float | None]) -> None:
    """Top status bar: weight, goal, loss, start, protocol days."""
    st.title(f"🏋️ {APP_TITLE}")
    st.caption(
        "Peptide protocol & fitness optimization — local, private, offline-first."
    )

    current = stats["current"]
    total_lost = stats["total_lost"]
    to_goal = (current - GOAL_WEIGHT_LBS) if current is not None else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Current weight",
        f"{current:.1f} lbs" if current is not None else "—",
        delta=f"{-total_lost:.1f} lbs" if total_lost is not None else None,
        delta_color="inverse",
    )
    c2.metric("Goal weight", f"{GOAL_WEIGHT_LBS:.0f} lbs")
    c3.metric(
        "Total lost",
        f"{total_lost:.1f} lbs" if total_lost is not None else "—",
    )
    c4.metric("Start weight", f"{START_WEIGHT_LBS:.0f} lbs")
    c5.metric("Days on protocol", str(days_on_protocol()))

    if to_goal is not None:
        remaining = max(to_goal, 0.0)
        progress = min(
            max((START_WEIGHT_LBS - current) / (START_WEIGHT_LBS - GOAL_WEIGHT_LBS), 0.0),
            1.0,
        )
        st.progress(
            progress,
            text=(
                f"{remaining:.1f} lbs to goal · "
                f"{progress * 100:.0f}% of journey complete"
            ),
        )


def render_current_stack() -> None:
    """Always-visible protocol stack card."""
    st.subheader("Current Stack")
    st.caption("Active protocol — no longer using CJC / Ipamorelin.")

    rows = []
    for item in CURRENT_STACK:
        status_label = "● Active" if item["status"] == "Active" else "○ Off"
        rows.append(
            {
                "Peptide": item["peptide"],
                "Dose": item["dose"],
                "Schedule": item["schedule"],
                "Status": status_label,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def render_daily_log_form() -> None:
    """Daily check-in form → SQLite."""
    st.subheader("Daily Log")
    st.caption("Log weight and subjective metrics. Saving the same date updates it.")

    with st.form("daily_log_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            log_date = st.date_input("Date", value=date.today())
            weight = st.number_input(
                "Weight (lbs)",
                min_value=50.0,
                max_value=500.0,
                value=float(get_latest_weight() or START_WEIGHT_LBS),
                step=0.1,
                format="%.1f",
            )
            energy = st.slider("Energy", 1, 10, 7)
            sleep = st.slider("Sleep quality", 1, 10, 7)
        with col_b:
            hunger = st.slider("Hunger / cravings", 1, 10, 4)
            joint_pain = st.slider("Joint pain", 1, 10, 2)
            alcohol = st.toggle("Alcohol today?", value=False)
            alcohol_amt = st.text_input(
                "Alcohol amount / type (if yes)",
                placeholder="e.g. 2 beers, 1 glass of wine",
            )
            notes = st.text_area("Notes", placeholder="Optional free-text notes…", height=100)

        submitted = st.form_submit_button("Save daily log", type="primary", use_container_width=True)

        if submitted:
            upsert_daily_log(
                {
                    "log_date": log_date.isoformat(),
                    "weight_lbs": float(weight),
                    "energy": int(energy),
                    "sleep": int(sleep),
                    "hunger": int(hunger),
                    "joint_pain": int(joint_pain),
                    "alcohol": 1 if alcohol else 0,
                    "alcohol_amt": alcohol_amt.strip() if alcohol else None,
                    "notes": notes.strip() or None,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            st.success(f"Saved daily log for {log_date.isoformat()}.")
            st.rerun()


def render_dose_log() -> None:
    """Log peptide doses and show last 14 days of history."""
    st.subheader("Dose Log")
    st.caption("Record exact units taken. One entry per peptide per day (updates on re-save).")

    # Peptide outside the form so default units refresh when selection changes
    defaults = {
        "Retatrutide": 45.0,
        "KLOW": 13.0,
        "Tesamorelin": 9.5,
        "Sermorelin": 12.0,
    }
    peptide = st.selectbox("Peptide", PEPTIDE_NAMES, key="dose_peptide")
    # Key includes peptide so the default units value resets on selection change
    units_key = f"dose_units_{peptide}"

    with st.form("dose_log_form", clear_on_submit=False):
        d1, d2 = st.columns(2)
        with d1:
            dose_date = st.date_input("Dose date", value=date.today(), key="dose_date")
            units = st.number_input(
                "Units taken",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults.get(peptide, 10.0)),
                step=0.5,
                format="%.1f",
                key=units_key,
            )
        with d2:
            dose_notes = st.text_input("Notes (optional)", placeholder="Site, time, etc.")

        dose_submitted = st.form_submit_button(
            "Save dose", type="primary", use_container_width=True
        )
        if dose_submitted:
            upsert_dose_log(
                {
                    "log_date": dose_date.isoformat(),
                    "peptide": peptide,
                    "units": float(units),
                    "notes": dose_notes.strip() or None,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            st.success(f"Logged {units:g} units of {peptide} on {dose_date.isoformat()}.")
            st.rerun()

    st.markdown("##### Last 14 days")
    dose_df = load_dose_logs(days=14)
    if dose_df.empty:
        st.info("No doses logged in the last 14 days yet.")
        return

    # Pivot for a clear day × peptide matrix
    display = dose_df.copy()
    display["log_date"] = display["log_date"].astype(str)
    pivot = display.pivot_table(
        index="log_date",
        columns="peptide",
        values="units",
        aggfunc="first",
    )
    # Ensure all peptides appear as columns
    for p in PEPTIDE_NAMES:
        if p not in pivot.columns:
            pivot[p] = pd.NA
    pivot = pivot[PEPTIDE_NAMES].sort_index(ascending=False)
    pivot.index.name = "Date"
    st.dataframe(
        pivot,
        use_container_width=True,
    )

    with st.expander("Raw dose history (last 14 days)"):
        st.dataframe(
            dose_df[["log_date", "peptide", "units", "notes"]].rename(
                columns={
                    "log_date": "Date",
                    "peptide": "Peptide",
                    "units": "Units",
                    "notes": "Notes",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_reconstitution() -> None:
    """Insulin-syringe reconstitution calculator with known presets."""
    st.subheader("Reconstitution Calculator")
    st.caption(
        "U-100 insulin syringe math: 1 mL = 100 units. "
        "Preloads match known vials (Reta 24 mg/2 mL → 12 mg/mL, Tesa 22 mg/2 mL → 11 mg/mL)."
    )

    preset_name = st.selectbox("Preset", list(RECON_PRESETS.keys()), key="recon_preset")
    preset = RECON_PRESETS[preset_name]

    # When the preset changes, push known vial/BAC values into widget state
    if st.session_state.get("_last_recon_preset") != preset_name:
        st.session_state["recon_vial"] = float(preset["vial_mg"])
        st.session_state["recon_bac"] = float(preset["bac_ml"])
        st.session_state["_last_recon_preset"] = preset_name

    r1, r2, r3 = st.columns(3)
    with r1:
        vial_mg = st.number_input(
            "Vial size (mg)",
            min_value=0.1,
            max_value=1000.0,
            step=0.5,
            format="%.1f",
            key="recon_vial",
        )
    with r2:
        bac_ml = st.number_input(
            "BAC water (mL)",
            min_value=0.1,
            max_value=20.0,
            step=0.1,
            format="%.1f",
            key="recon_bac",
        )
    with r3:
        dose_unit = st.selectbox("Dose unit", ["mcg", "mg"], index=0)

    desired = st.number_input(
        f"Desired dose ({dose_unit})",
        min_value=0.001,
        max_value=10000.0 if dose_unit == "mcg" else 50.0,
        value=1000.0 if dose_unit == "mcg" else 1.0,
        step=10.0 if dose_unit == "mcg" else 0.1,
        format="%.3f" if dose_unit == "mg" else "%.1f",
        key="recon_dose",
    )

    try:
        result = calc_reconstitution(vial_mg, bac_ml, desired, dose_unit)
    except ValueError as exc:
        st.error(str(exc))
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Concentration", f"{result['concentration_mg_ml']:.3f} mg/mL")
    m2.metric("Dose", f"{result['dose_mg']:.4g} mg")
    m3.metric("Draw volume", f"{result['volume_ml']:.4f} mL")
    m4.metric("Syringe units", f"{result['units']:.1f} U")

    st.info(
        f"**{result['mcg_per_unit']:.1f} mcg per unit** on a U-100 syringe · "
        f"For {desired:g} {dose_unit}, draw **{result['units']:.1f} units**."
    )

    # Quick reference table for common unit draws
    with st.expander("Quick reference — units → dose at this concentration"):
        unit_ticks = [5, 10, 12, 13, 20, 30, 40, 45, 50, 60, 80, 100]
        rows = []
        for u in unit_ticks:
            mg = (u / 100.0) * result["concentration_mg_ml"]
            rows.append(
                {
                    "Units": u,
                    "mg": round(mg, 4),
                    "mcg": round(mg * 1000.0, 1),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_progress(df: pd.DataFrame, stats: dict[str, float | None]) -> None:
    """Weight trend chart + summary stats."""
    st.subheader("Progress")

    total_lost = stats["total_lost"]
    avg_weekly = stats["avg_weekly"]
    current = stats["current"]

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Total lost",
        f"{total_lost:.1f} lbs" if total_lost is not None else "—",
    )
    s2.metric(
        "Avg weekly loss",
        f"{avg_weekly:.2f} lbs/wk" if avg_weekly is not None else "—",
    )
    remaining = (current - GOAL_WEIGHT_LBS) if current is not None else None
    s3.metric(
        "Remaining to goal",
        f"{remaining:.1f} lbs" if remaining is not None else "—",
    )

    weight_df = (
        df.dropna(subset=["weight_lbs"]).sort_values("log_date")
        if not df.empty
        else df
    )
    if weight_df.empty:
        st.info("Log a weight to see your progress chart.")
        return

    chart_df = weight_df.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["log_date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Date"],
            y=chart_df["weight_lbs"],
            mode="lines+markers",
            name="Weight",
            line=dict(color="#6C63FF", width=3),
            marker=dict(size=8, color="#8B85FF"),
            hovertemplate="%{x|%b %d}: %{y:.1f} lbs<extra></extra>",
        )
    )
    # Goal reference line
    fig.add_hline(
        y=GOAL_WEIGHT_LBS,
        line_dash="dash",
        line_color="#3DDC97",
        annotation_text=f"Goal {GOAL_WEIGHT_LBS:.0f} lbs",
        annotation_position="top left",
        annotation_font_color="#3DDC97",
    )
    # Start reference
    fig.add_hline(
        y=START_WEIGHT_LBS,
        line_dash="dot",
        line_color="#9AA3B5",
        annotation_text=f"Start {START_WEIGHT_LBS:.0f} lbs",
        annotation_position="bottom left",
        annotation_font_color="#9AA3B5",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Weight over time",
        xaxis={**AXIS_STYLE, "title": None},
        yaxis={**AXIS_STYLE, "title": "Weight (lbs)"},
        height=380,
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Secondary wellness trends if data exists
    wellness_cols = ["energy", "sleep", "hunger", "joint_pain"]
    if all(c in weight_df.columns for c in wellness_cols):
        with st.expander("Wellness trends (energy, sleep, hunger, joint pain)"):
            long = weight_df.melt(
                id_vars=["log_date"],
                value_vars=wellness_cols,
                var_name="Metric",
                value_name="Score",
            )
            long["log_date"] = pd.to_datetime(long["log_date"])
            long["Metric"] = long["Metric"].replace(
                {
                    "energy": "Energy",
                    "sleep": "Sleep",
                    "hunger": "Hunger",
                    "joint_pain": "Joint pain",
                }
            )
            wfig = px.line(
                long,
                x="log_date",
                y="Score",
                color="Metric",
                markers=True,
                labels={"log_date": "Date", "Score": "Score (1–10)"},
            )
            # Do not pass yaxis twice — merge range into a single yaxis dict
            wfig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis={**AXIS_STYLE},
                yaxis={**AXIS_STYLE, "range": [0, 11]},
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(wfig, use_container_width=True)


def render_history_and_export(df: pd.DataFrame) -> None:
    """Recent daily logs + CSV export tools."""
    st.subheader("History & Export")

    if df.empty:
        st.info("No daily logs yet. Your data will appear here after the first save.")
    else:
        show = df.sort_values("log_date", ascending=False).copy()
        show["alcohol"] = show["alcohol"].map({1: "Yes", 0: "No"})
        show = show.rename(
            columns={
                "log_date": "Date",
                "weight_lbs": "Weight (lbs)",
                "energy": "Energy",
                "sleep": "Sleep",
                "hunger": "Hunger",
                "joint_pain": "Joint pain",
                "alcohol": "Alcohol",
                "alcohol_amt": "Alcohol detail",
                "notes": "Notes",
            }
        )
        cols = [
            "Date",
            "Weight (lbs)",
            "Energy",
            "Sleep",
            "Hunger",
            "Joint pain",
            "Alcohol",
            "Alcohol detail",
            "Notes",
        ]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download daily logs (CSV)",
            data=export_table_csv("daily_logs"),
            file_name=f"pep_rally_daily_{date.today().isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Download dose logs (CSV)",
            data=export_table_csv("dose_logs"),
            file_name=f"pep_rally_doses_{date.today().isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(f"Database file: `{DB_PATH}`")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_db()

    daily_df = load_daily_logs()
    stats = compute_weight_stats(daily_df)

    render_header(stats)
    st.divider()
    render_current_stack()
    st.divider()

    # Daily log + dose log side by side on wide screens
    left, right = st.columns(2)
    with left:
        render_daily_log_form()
    with right:
        render_dose_log()

    st.divider()
    render_reconstitution()
    st.divider()
    render_progress(daily_df, stats)
    st.divider()
    render_history_and_export(daily_df)

    st.markdown(
        "<div style='text-align:center;color:#6B7280;font-size:0.8rem;"
        "margin-top:1.5rem;'>PEP RALLY Tracker · local SQLite · your data stays on this machine</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
