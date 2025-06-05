# app.py  ·  Google Places (New) Enricher + Map  — v2025-06-04-revB
# ──────────────────────────────────────────────────────────────────
"""
Key features
============
• Text-Search + automatic second-pass Place-Details (default, no nearby API)
• Optional direct Place-Details mode
• Interactive Folium map of all results (Streamlit-embedded)
• Friendly output column names (Name, Address, Website, Phone, …)
"""

from __future__ import annotations

import io, os, re, time, json, requests, urllib.parse as urlp
from typing import List


import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ── persistent session storage ─────────────────────────────────────
if "df_out" not in st.session_state:     # DataFrame with results
    st.session_state["df_out"] = None
    st.session_state["df_ready"] = False # flag: have results to show?

# ═════════════════════  SMALL HELPERS  ════════════════════════════

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique – required by Streamlit’s Arrow grid."""
    seen, fresh = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            fresh.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            fresh.append(c)
    df.columns = fresh
    return df


def safe_df(df: pd.DataFrame):
    """Show DataFrame but fall back to st.table on rare JS-grid hiccups."""
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)


def show_map(df: pd.DataFrame):
    """Embed an interactive Folium map if Latitude/Longitude are present."""
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df = df.copy()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"])
        if df.empty:
            st.info("No valid coordinates to plot.")
            return

        fmap = folium.Map(
            location=[df["Latitude"].mean(), df["Longitude"].mean()],
            zoom_start=6,
            tiles="cartodbpositron",
        )
        for _, r in df.iterrows():
            folium.CircleMarker(
                [r["Latitude"], r["Longitude"]],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.6,
                popup=str(r.get("FirmName") or r.get("Name") or ""),
            ).add_to(fmap)

        st_folium(fmap, height=500, use_container_width=True)

# ════════════════  THIN WRAPPER · PLACES API v1  ═══════════════════

PLACES_ROOT = "https://places.googleapis.com/v1"
HEADERS_POST = {"Content-Type": "application/json; charset=utf-8"}


class PlacesV1:
    """Ultra-thin, rate-limited wrapper around REST endpoints."""

    def __init__(self, key: str, qps: float):
        if not key:
            raise ValueError("⚠️  API key cannot be blank – paste it in the sidebar.")
        self.key = key
        self.qps = qps
        self._last_call = 0.0

    # ---- low-level helpers ---------------------------------------
    def _throttle(self):
        min_gap = max(0.1, 1 / self.qps)          # ≥100 ms between calls
        lag = min_gap - (time.time() - self._last_call)
        if lag > 0:
            time.sleep(lag)
        self._last_call = time.time()

    def _post(self, route: str, body: dict, mask: str):
        self._throttle()
        if not mask:
            raise ValueError("Field mask cannot be empty – select at least one field.")
        url = f"{PLACES_ROOT}{route}?key={self.key}"
        hdrs = HEADERS_POST.copy()
        hdrs["X-Goog-FieldMask"] = mask
        r = requests.post(url, headers=hdrs, json=body, timeout=30)
        if not r.ok:
            raise RuntimeError(f"{r.status_code} – {r.text[:300]}")
        return r.json()

    def _details(self, rid: str, mask: str):
        self._throttle()
        if not mask:
            raise ValueError("Field mask cannot be empty – select at least one field.")
        rid = rid.lstrip("/")
        path = rid if rid.startswith("places/") else f"places/{rid}"
        params = urlp.urlencode({"key": self.key, "fields": mask}, safe=",")
        url = f"{PLACES_ROOT}/{path}?{params}"
        r = requests.get(url, timeout=30)
        if not r.ok:
            raise RuntimeError(f"{r.status_code} – {r.text[:300]}")
        return r.json()

    # ---- public shortcuts ----------------------------------------
    def text(self, body, mask):
        return self._post("/places:searchText", body, mask)

    def details(self, rid, mask):
        return self._details(rid, mask)

# ═════════════════════  STREAMLIT UI  ══════════════════════════════

st.set_page_config(page_title="Google Places Enricher", layout="wide")
st.title("🗺️ Google Places (New) Data Enricher")

# ── sidebar ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 API credentials")
    API_KEY = st.text_input("Google API key", type="password")
    QPS = st.number_input("Queries / second", 0.1, 60.0, 5.0)

    st.header("⚙️ Pick endpoint")
    MODE = st.radio("Service", ("Text Search", "Place Details"))

    st.header("🧩 Field mask (must pick at least 1)")
    DEFAULT_MASK: List[str] = [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.websiteUri",
        "places.businessStatus",
        "places.location",
        "places.internationalPhoneNumber",
        "places.types",
        "places.googleMapsUri",
    ]
    MASK = st.multiselect("Return fields", options=DEFAULT_MASK, default=DEFAULT_MASK)

    if MODE == "Text Search":
        st.header("🔤 Text-search options")
        LANG = st.text_input("languageCode (optional)")
        REGION = st.text_input("regionCode (optional)")

        # ── advanced tuning (optional) ───────────────────────────────
        st.header("🔍 Improve matching")
        MAX_RES = st.slider("Max results per query", 1, 60, 20)
        RANK    = st.selectbox(
            "rankPreference", ("POPULARITY", "DISTANCE"),
            help="See Places Text Search docs for ranking behaviour",
        )
        BIAS = st.text_input(
            "locationBias (circle or rectangle)",
            help="Example: circle:10000@52.52,13.40",
        )
        RESTRICT = st.text_input(
            "locationRestriction (rectangle)",
            help="Example: rectangle:12.90,55.30|15.04,57.75",
        )
    else:  # ── Place Details mode ───────────────────────────────────
        COL_PID = st.text_input("Column with place_id or resource", "place_id")

        # dummy defaults so code outside the sidebar always finds variables
        MAX_RES  = 1
        RANK     = "POPULARITY"
        BIAS     = ""
        RESTRICT = ""
        LANG     = ""
        REGION   = ""

    RUN = st.button("🚀 Enrich", disabled=not API_KEY)

# ── file upload -----------------------------------------------------
st.subheader("1️⃣ Upload Excel or CSV")
up_file = st.file_uploader("Choose file", type=["xlsx", "xls", "csv"])

@st.cache_data(show_spinner=False)
def read_df(f):
    ext = os.path.splitext(f.name)[1].lower()
    return pd.read_excel(f) if ext in {".xlsx", ".xls"} else pd.read_csv(f)

if not up_file:
    st.stop()

DF_IN = read_df(up_file)
safe_df(DF_IN.head())

# ── pick query columns (Text mode) ---------------------------------
if MODE == "Text Search":
    st.subheader("2️⃣ Select query column(s)")
    COLS = st.multiselect("Columns (joined with spaces)", DF_IN.columns.tolist())
    if not COLS:
        st.warning("Select at least one column.")
        st.stop()
else:
    COLS = []  # type: ignore

# ═══════════════  ENRICHMENT ENGINE  ═══════════════════════════════

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    api = PlacesV1(API_KEY, QPS)
    mask_search = ",".join(MASK)
    mask_det = ",".join(re.sub(r"^places\.", "", f) for f in MASK)

    def join(r):  # combine selected cols into a single query string
        return " ".join(str(r[c]).strip() for c in COLS if pd.notna(r[c]) and str(r[c]).strip())

    # ---- per-row handlers ----------------------------------------
    def _text(r):
        """Text-Search ➜ (optionally) second-pass details ➜ list of hits."""
        q = join(r)
        if not q:
            return []

        # ---- first request -------------------------------------------------
        body = {
            "textQuery": q,
            "pageSize"  : min(20, MAX_RES),   # Google returns ≤20 per page
            "rankPreference": RANK,
        }
        if LANG:
            body["languageCode"] = LANG
        if REGION:
            body["regionCode"] = REGION
        if BIAS:
            body["locationBias"] = BIAS
        if RESTRICT:
            body["locationRestriction"] = RESTRICT

        places, token = [], None
        resp = api.text(body, mask_search)
        places.extend(resp.get("places", []))
        token = resp.get("nextPageToken")

        # ---- follow nextPageToken up to Google’s 60-result cap -------------
        while token and len(places) < MAX_RES:
            time.sleep(2)                                # 2-sec rule
            resp   = api.text({"pageToken": token}, mask_search)
            places.extend(resp.get("places", []))
            token  = resp.get("nextPageToken")

        # ---- mandatory second-pass details for every hit ------------------
        detailed = []
        for p in places[:MAX_RES]:
            detailed.append(api.details(p["id"], mask_det))
        return detailed          # always a list (possibly empty)


    def _details(r):
        rid = r.get(COL_PID) or r.get("name", "").split("/")[-1]
        return api.details(rid, mask_det) if rid else None

    H = _text if MODE == "Text Search" else _details

    out_rows = []
    bar = st.progress(0.0, "Calling API…")

    for i, row in df.iterrows():
        try:
            hits = H(row)             # ← now a *list* (0–60)
            for h in hits:
                if h:
                    h["__src_row"] = i     # keep link to original row
                    out_rows.append(h)
        except Exception as e:
            st.error(f"Row {i}: {e}")
        bar.progress((i + 1) / len(df))

    bar.empty()

    # --- turn place hits into a DataFrame --------------------------------
    flat = pd.json_normalize(out_rows)              # 0-to-many rows

    # --- replicate each input row for every hit (1 : m join) -------------
    base = df.reset_index().rename(
        columns={"index": "__src_row"}              # unique key
    )

    full = base.merge(flat, on="__src_row", how="right")

    # optional: keep original order and tidy up
    full.sort_values("__src_row", inplace=True)
    full.drop(columns="__src_row", inplace=True)
    full.reset_index(drop=True, inplace=True)

    # ---- tidy & rename columns -----------------------------------
    if "location.latitude" in full.columns:
        full["Latitude"] = full.pop("location.latitude")
    if "location.longitude" in full.columns:
        full["Longitude"] = full.pop("location.longitude")

    friendly = {
        "displayName.text": "Name",
        "formattedAddress": "Address",
        "websiteUri": "Website",
        "businessStatus": "Status",
        "internationalPhoneNumber": "Phone",
        "googleMapsUri": "GoogleMapsURL",
    }
    full.rename(columns={k: v for k, v in friendly.items() if k in full.columns},
                inplace=True)

    return dedup_columns(full)

# ── run & UI --------------------------------------------------------
# ── 3 · run enrichment only when the button is clicked -------------
if RUN:
    with st.spinner("Enriching…"):
        DF_OUT = enrich(DF_IN)

    # 🔑 save once for future reruns
    st.session_state["df_out"] = DF_OUT
    st.session_state["df_ready"] = True

# ── 4 · always show last results if they exist ---------------------
if st.session_state.get("df_ready"):
    DF_OUT = st.session_state["df_out"]

    st.success(f"✅ {len(DF_OUT)} rows enriched.")
    safe_df(DF_OUT)
    show_map(DF_OUT)              # map keeps re-rendering

    # ---- download buttons ----------------------------------------
    def xlsx_bytes(df):
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        return buf.getvalue()

    c1, c2 = st.columns(2)
    c1.download_button(
        "💾 Excel",
        xlsx_bytes(DF_OUT),
        file_name="places_enriched.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    c2.download_button(
        "💾 CSV",
        DF_OUT.to_csv(index=False).encode(),
        file_name="places_enriched.csv",
        mime="text/csv",
    )

# ── usage tips ------------------------------------------------------
if not RUN:
    st.markdown(
        """
        **📄 Usage notes**
        1. The *field mask* is required by Google – leave at least one box ticked.
        2. For *Place Details* mode, your sheet must include either  
           a `place_id` column **or** a full resource like `places/ChIJ…`.
        3. If you get **403 PERMISSION_DENIED**, enable the Places API (New) in
           your Google Cloud project and check key restrictions.
        """
    )
