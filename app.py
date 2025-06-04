# app.py  ·  Google Places (New) Enricher  — v2025‑06‑04‑revB
# ─────────────────────────────────────────────────────────────────────
"""
Streamlined version
------------------
✓ Removed **Nearby Search** – only **Text Search** (with automatic second‑pass
  *Place Details*) and **Place Details** endpoints remain.
✓ Second‑pass Place‑Details now *always* runs for every Text‑Search hit.
✓ Added user‑friendly column names in the final download.
✓ Minor UI polish – simpler sidebar without the extra checkbox.
"""
from __future__ import annotations

import io
import os
import re
import time
import requests
import urllib.parse as urlp
from typing import List

import pandas as pd
import streamlit as st

# ═════════════════════════  SMALL HELPERS  ═══════════════════════════

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique – Arrow/Streamlit requires this."""
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
    """Show a DataFrame but fall back to st.table on rare JS‑grid hiccups."""
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)


# ════════════════  THIN WRAPPER · PLACES API v1  ════════════════════

PLACES_ROOT = "https://places.googleapis.com/v1"
HEADERS_POST = {"Content-Type": "application/json; charset=utf-8"}


class PlacesV1:
    """Ultra‑thin, rate‑limited wrapper around the REST endpoints."""

    def __init__(self, key: str, qps: float):
        if not key:
            raise ValueError("⚠️  API key cannot be blank – paste it in the sidebar.")
        self.key = key
        self.qps = qps
        self._last_call = 0.0

    # ---------- low‑level helpers -----------------------------------
    def _throttle(self):
        """Respect the user‑set QPS *and* sleep a minimum of 100 ms."""
        min_gap = max(0.1, 1 / self.qps)  # at least 100 ms between calls
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

    # ---------- public shortcuts ------------------------------------
    def text(self, body, mask):
        return self._post("/places:searchText", body, mask)

    def details(self, rid, mask):
        return self._details(rid, mask)


# ════════════════════  STREAMLIT INTERFACE  ═════════════════════════

st.set_page_config(page_title="Google Places Enricher", layout="wide")
st.title("🗺️ Google Places (New) Data Enricher")

# ── sidebar controls -------------------------------------------------
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

    # endpoint‑specific widgets
    if MODE == "Text Search":
        st.header("🔤 Text‑search options")
        LANG = st.text_input("languageCode (optional)")
        REGION = st.text_input("regionCode (optional)")
    else:  # Place Details
        COL_PID = st.text_input("Column with place_id or resource", "place_id")

    RUN = st.button("🚀 Enrich", disabled=not API_KEY)

# ── file upload only (no paste) --------------------------------------
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

# ── pick query columns when needed -----------------------------------
if MODE == "Text Search":
    st.subheader("2️⃣ Select query column(s)")
    COLS = st.multiselect("Columns (joined with spaces)", DF_IN.columns.tolist())
    if not COLS:
        st.warning("Select at least one column.")
        st.stop()
else:
    COLS = []  # type: ignore

# ═══════════════════  ENRICHMENT ENGINE  ════════════════════════════

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    api = PlacesV1(API_KEY, QPS)
    mask_search = ",".join(MASK)
    mask_det = ",".join(re.sub(r"^places\.", "", f) for f in MASK)

    def join(r):
        return " ".join(
            str(r[c]).strip() for c in COLS if pd.notna(r[c]) and str(r[c]).strip()
        )

    # ---------- per‑row handlers ------------------------------------
    def _text(r):
        q = join(r)
        if not q:
            return None
        body = {"textQuery": q}
        if LANG:
            body["languageCode"] = LANG
        if REGION:
            body["regionCode"] = REGION
        # 1️⃣ text search
        res = api.text(body, mask_search)
        hit = res.get("places", [None])[0]
        if not hit:
            return None
        # 2️⃣ always do details lookup
        return api.details(hit["id"], mask_det)

    def _details(r):
        rid = r.get(COL_PID) or r.get("name", "").split("/")[-1]
        return api.details(rid, mask_det) if rid else None

    H = {"Text Search": _text, "Place Details": _details}[MODE]

    out_rows = []
    bar = st.progress(0.0, "Calling API…")
    for i, row in df.iterrows():
        try:
            out_rows.append(H(row))
        except Exception as e:
            st.error(f"Row {i}: {e}")
            out_rows.append(None)
        bar.progress((i + 1) / len(df))
    bar.empty()

    flat = pd.json_normalize(out_rows)
    full = pd.concat([df.reset_index(drop=True), flat], axis=1)

    # friendly column names ----------------------------------------------------
    rename_map = {
        "displayName.text": "Name",
        "formattedAddress": "Address",
        "websiteUri": "Website",
        "businessStatus": "Business Status",
        "internationalPhoneNumber": "Phone",
        "googleMapsUri": "Google Maps URL",
    }
    full.rename(columns=rename_map, inplace=True)

    # flatten location → Latitude / Longitude
    if "location.latitude" in full.columns:
        full["Latitude"] = full.pop("location.latitude")
    if "location.longitude" in full.columns:
        full["Longitude"] = full.pop("location.longitude")

    # tidy types list into string if present
    if "types" in full.columns:
        full["Types"] = full["types"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        full.drop("types", axis=1, inplace=True)

    return dedup_columns(full)


# ── run & download ---------------------------------------------------
if RUN:
    with st.spinner("Enriching…"):
        DF_OUT = enrich(DF_IN)
    st.success(f"✅ {len(DF_OUT)} rows enriched.")
    safe_df(DF_OUT.head())

    # -------- download helpers --------------------------------------
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

# ── usage tips -------------------------------------------------------
if not RUN:
    st.markdown(
        """
        **📄 Usage notes**
        1. The *field mask* is required by Google – leave at least one box ticked.
        2. For *Place Details* mode, make sure your sheet contains either a
           `place_id` column **or** a `name` column with full resources in the
           form `places/ChIJ…`.
        3. Errors such as **403 PERMISSION_DENIED** usually mean the Places API
           (New) is not yet enabled on your Google Cloud project.
        """
    )
