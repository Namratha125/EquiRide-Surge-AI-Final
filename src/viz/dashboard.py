# src/viz/dashboard.py
import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path
import h3
import os
import time

# --- 1. FILE PATHS ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"
FORECAST_CSV = DATA_DIR / "forecast_15min_predictions.csv"
REPOSITION_CSV = DATA_DIR / "repositioning_plan.csv"
PROCESSED_DATA_CSV = DATA_DIR / "processed_data.csv"
ALERTS_CSV = DATA_DIR / "surge_alerts.csv"
CSS_PATH = Path(__file__).resolve().parent / "styles.css"

# --- 2. CREDENTIALS ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACd16431c489770b29a82b76faa8e98b52")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "") # Will load from env if set
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+14784006602")
DEFAULT_ALERT_PHONE = os.getenv("ALERT_PHONE_NUMBER", "+919611751505")

def local_css(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _normalize_columns(df: pd.DataFrame):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# --- 3. DATA LOADING ---
@st.cache_data
def load_coordinate_mapping(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    
    h3_col = next((c for c in df.columns if c in ("h3_index", "h3", "hex_id")), None)
    lat_col = next((c for c in df.columns if c in ("latitude", "lat")), None)
    lon_col = next((c for c in df.columns if c in ("longitude", "lon", "lng")), None)
    zone_col = next((c for c in df.columns if c in ("area_name", "area", "zone_name", "name")), None)

    if not (h3_col and lat_col and lon_col): return pd.DataFrame()

    cols = [h3_col, lat_col, lon_col]
    if zone_col: cols.append(zone_col)
        
    coords_df = df[cols].drop_duplicates(subset=[h3_col]).copy()
    coords_df = coords_df.rename(columns={h3_col: "h3_index", lat_col: "lat", lon_col: "lon"})
    if zone_col: coords_df = coords_df.rename(columns={zone_col: "zone_name"})
    else: coords_df["zone_name"] = coords_df["h3_index"].astype(str)
    return coords_df

@st.cache_data
def load_forecasts(forecast_path: Path) -> pd.DataFrame:
    if not forecast_path.exists(): return pd.DataFrame()
    df = pd.read_csv(forecast_path)
    df = _normalize_columns(df)

    h3_col = next((c for c in df.columns if c in ("h3_index", "h3")), None)
    score_col = next((c for c in df.columns if c in ("pred_bookings_15min", "pred_bookings", "pred", "surge_score")), None)
    time_col = next((c for c in df.columns if c in ("next_time","timestamp","time")), None)

    if not (h3_col and score_col and time_col): return pd.DataFrame()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.rename(columns={h3_col: "h3", score_col: "surge_score", time_col: "timestamp"})
    df = df.dropna(subset=["h3", "timestamp"])
    df["surge_score"] = pd.to_numeric(df["surge_score"], errors="coerce").fillna(0.0)
    return df

def merge_forecast_with_coords(forecast_df, coords_df, selected_ts):
    if forecast_df.empty or coords_df.empty: return pd.DataFrame()
    # Match without microseconds for safer slider interaction
    subset = forecast_df[forecast_df["timestamp"].dt.floor('S') == selected_ts.floor('S')].copy()
    
    if subset.empty:
        subset = forecast_df.copy() # Fallback

    merged = pd.merge(subset, coords_df, left_on="h3", right_on="h3_index", how="left")
    merged = merged.dropna(subset=["lat", "lon"])
    
    max_score = merged["surge_score"].max() if not merged.empty else 0
    if max_score > 0:
        merged["surge_score"] = merged["surge_score"] / max_score
    return merged

def load_reposition(reposition_path, coords_df):
    if not reposition_path.exists(): return []
    df = pd.read_csv(reposition_path)
    df = _normalize_columns(df)
    
    # --- FIX: Standardize Column Names ---
    # The CSV has 'estimated_profit' but the dashboard logic expects 'profit_est'
    rename_map = {
        "estimated_profit": "profit_est",
        "repositioning_distance_km": "distance_km",
        "assigned_zone": "assigned_zone",
        "assigned_h3": "assigned_zone"
    }
    df = df.rename(columns=rename_map)
    # -------------------------------------

    assigned_col = next((c for c in df.columns if c in ("assigned_zone", "h3_index")), None)
    if not assigned_col: return []
        
    # Merge with coordinates to get destination Lat/Lon
    merged = pd.merge(df, coords_df, left_on=assigned_col, right_on="h3_index", how="left")
    merged = merged.rename(columns={"lat": "to_lat", "lon": "to_lon"})
    
    # Drop rows where we couldn't find coordinates for the target zone
    merged = merged.dropna(subset=["to_lat", "to_lon"])
    
    # Simulate driver starting positions if not provided
    if "from_lat" not in merged.columns:
        if not coords_df.empty and len(merged) <= len(coords_df):
            # Pick random start points from existing zones for realism
            rnd = coords_df.sample(n=len(merged), replace=True).reset_index(drop=True)
            merged["from_lat"] = rnd["lat"].values
            merged["from_lon"] = rnd["lon"].values
        else:
            # Fallback: Start slightly offset from destination
            merged["from_lat"] = merged["to_lat"] + 0.01
            merged["from_lon"] = merged["to_lon"] + 0.01
            
    return merged.to_dict(orient="records")

def load_alerts_for_map(alerts_path, coords_df):
    """Only used for map dots, not for the table."""
    if not alerts_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(alerts_path)
        df = _normalize_columns(df)
        if "h3_index" in df.columns and not coords_df.empty:
            if "severity" not in df.columns: df["severity"] = "HIGH"
            merged = pd.merge(df, coords_df, on="h3_index", how="left")
            return merged.dropna(subset=["lat", "lon"])
        return pd.DataFrame()
    except: return pd.DataFrame()

def h3_polygon_geojson(h3idx):
    try: return [[p[1], p[0]] for p in h3.h3_to_geo_boundary(str(h3idx), geo_json=True)]
    except: return []

# --- 4. APP LAYOUT ---
st.set_page_config(layout="wide", page_title="EquiRide Dashboard", initial_sidebar_state="expanded")
local_css(CSS_PATH)

st.markdown("""
    <div class="banner">
      <h1 style="margin:0; font-size:36px;">EquiRide — Surge Dashboard</h1>
      <div style="opacity:0.9">Real-time Monitoring • Driver Repositioning • Live Alerts</div>
    </div>
""", unsafe_allow_html=True)

# Load Data
coords_df = load_coordinate_mapping(PROCESSED_DATA_CSV)
forecast_df_raw = load_forecasts(FORECAST_CSV)
reposition_data = load_reposition(REPOSITION_CSV, coords_df)
reposition_df = pd.DataFrame(reposition_data)
alerts_map_df = load_alerts_for_map(ALERTS_CSV, coords_df)

# Time Logic
if forecast_df_raw.empty:
    st.error("No forecast data available. Run the pipeline first.")
    st.stop()

unique_times = sorted(pd.to_datetime(forecast_df_raw["timestamp"].dropna().unique()))
if len(unique_times) > 0:
    time_strings = [t.strftime("%H:%M:%S") for t in unique_times]
    # FIX FOR RANGE ERROR: Only show slider if >1 option
    if len(time_strings) > 1:
        selected_idx = st.sidebar.select_slider("Forecast Window", options=range(len(time_strings)), format_func=lambda x: time_strings[x])
        selected_ts = unique_times[selected_idx]
    else:
        selected_ts = unique_times[0]
        st.sidebar.info(f"Live Forecast: {time_strings[0]}")
    
    st.sidebar.caption(f"Date: {selected_ts.strftime('%Y-%m-%d')}")
    selected_df = merge_forecast_with_coords(forecast_df_raw, coords_df, selected_ts)
else:
    st.error("Forecast data is invalid (no timestamps).")
    st.stop()

# --- 5. MAP VISUALIZATION ---
layers = []
view_state = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=11, pitch=40)

if not selected_df.empty:
    mid = (float(selected_df.iloc[0]["lat"]), float(selected_df.iloc[0]["lon"]))
    view_state = pdk.ViewState(latitude=mid[0], longitude=mid[1], zoom=12, pitch=40)

    # Heatmap
    if selected_df["surge_score"].max() > 0:
        heat_data = [{"position": [r["lon"], r["lat"]], "weight": float(r["surge_score"])} for _, r in selected_df.iterrows()]
        layers.append(pdk.Layer("HeatmapLayer", data=heat_data, get_position="position", get_weight="weight", radiusPixels=80, opacity=0.6))

    # Hexagons
    hex_polys = []
    for _, r in selected_df.iterrows():
        poly = h3_polygon_geojson(r["h3"])
        if poly:
            hex_polys.append({"polygon": poly, "surge": float(r["surge_score"]), "name": r.get("zone_name", "Unknown")})
    
    if hex_polys:
        layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(hex_polys), get_polygon="polygon", get_fill_color="[255*surge, 60*(1-surge), 200*(1-surge), 100]", pickable=True, stroked=True, get_line_color=[50, 50, 50], line_width_min_pixels=1, auto_highlight=True))

# Arrows
if not reposition_df.empty:
    layers.append(pdk.Layer("ScatterplotLayer", data=reposition_df, get_position=["to_lon", "to_lat"], get_radius=80, get_fill_color=[0, 255, 0, 200]))
    path_data = [{"path": [[r["from_lon"], r["from_lat"]], [r["to_lon"], r["to_lat"]]]} for r in reposition_data]
    layers.append(pdk.Layer("PathLayer", data=path_data, get_path="path", width_scale=10, width_min_pixels=2, get_color=[50, 205, 50, 200]))

# Alert Dots
if not alerts_map_df.empty:
    critical = alerts_map_df[alerts_map_df["severity"].str.upper().isin(["CRITICAL", "HIGH"])]
    if not critical.empty:
        layers.append(pdk.Layer("ScatterplotLayer", data=critical, get_position=["lon", "lat"], get_radius=400, get_fill_color=[255, 0, 0, 180], stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=3, pickable=True))

deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"html": "<b>Zone:</b> {name}<br/><b>Surge:</b> {surge}"})
st.pydeck_chart(deck)

# --- 9. LOWER SECTION: FLEET OPS & DISPATCH ---
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🚀 Fleet Operations Center")
    
    # Check using the NEW standardized column names
    if not reposition_df.empty and "profit_est" in reposition_df.columns:
        # 1. KPI Metrics from Repositioning Plan
        total_drivers = len(reposition_df)
        total_profit = reposition_df["profit_est"].sum()
        avg_dist = reposition_df["distance_km"].mean() if "distance_km" in reposition_df.columns else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Drivers Mobilized", total_drivers, "+Active")
        k2.metric("Proj. Revenue Upside", f"₹{total_profit:,.0f}", "High Demand")
        k3.metric("Avg Dispatch Dist", f"{avg_dist:.1f} km", "Efficient")

        # 2. Chart
        st.caption("Top Target Zones by Profit Potential")
        if "zone_name" in reposition_df.columns:
            viz_df = reposition_df.groupby("zone_name")["profit_est"].sum().sort_values(ascending=False).head(7)
        else:
            viz_df = reposition_df.groupby("assigned_zone")["profit_est"].sum().sort_values(ascending=False).head(7)
        st.bar_chart(viz_df, color="#00CC00") 
    else:
        st.info("No active repositioning plan. Fleet is stationary.")

with col2:
    st.subheader("📲 Dispatch Console")
    with st.form("sms_form"):
        dest_phone = st.text_input("Driver Phone", value=DEFAULT_ALERT_PHONE)
        default_msg = "ALERT: High demand surge detected."
        if not reposition_df.empty:
            top_zone = reposition_df.iloc[0]['assigned_zone']
            if not coords_df.empty:
                name_match = coords_df[coords_df['h3_index'] == top_zone]['zone_name']
                if not name_match.empty:
                    top_zone = name_match.values[0]
            default_msg = f"DISPATCH: Go to {top_zone}. High fare opportunity detected."
        msg_body = st.text_area("Message", value=default_msg, height=100)
        
        if st.form_submit_button("🚀 Send Instructions"):
            if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
                st.error("Missing Twilio Credentials.")
            else:
                try:
                    from twilio.rest import Client
                    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                    m = client.messages.create(body=msg_body, from_=TWILIO_PHONE_NUMBER, to=dest_phone)
                    st.success(f"Sent! ID: {m.sid}")
                except Exception as e:
                    st.error(str(e))

# --- 10. SIDEBAR & SIMULATION ---
st.sidebar.header("Zone Analysis")
if not selected_df.empty and "zone_name" in selected_df.columns:
    zones = sorted(list(selected_df["zone_name"].dropna().unique()))
    if zones:
        ch = st.sidebar.selectbox("Inspect Zone", zones)
        row = selected_df[selected_df["zone_name"] == ch].iloc[0]
        st.sidebar.metric("Surge Score", f"{row['surge_score']:.2f}")
        st.sidebar.caption(f"H3: {row['h3']}")

if not reposition_df.empty:
    st.markdown("---")
    if st.button("▶️ Run Simulation"):
        ph = st.empty()
        for s in range(1, 21):
            r = s/20
            dots = [{"lon": d['from_lon'] + (d['to_lon']-d['from_lon'])*r, "lat": d['from_lat'] + (d['to_lat']-d['from_lat'])*r} for d in reposition_data]
            sl = pdk.Layer("ScatterplotLayer", data=pd.DataFrame(dots), get_position=["lon", "lat"], get_radius=120, get_fill_color=[255, 215, 0, 255])
            ph.pydeck_chart(pdk.Deck(layers=layers+[sl], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10"))
            time.sleep(0.04)
