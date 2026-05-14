import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import html
import os as pyos
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from inference import SearchEngine as FullSearchEngine
from inference_speed import SearchEngine as FastSearchEngine
from ltrpkg.config import get_settings
from ltrpkg.utils import get_logger, process_memory_mb, safe_percentile, tail_log


BASE_DIR = Path(__file__).resolve().parent
DEMO_WEB_DIR = BASE_DIR / "data/web"
UPLOADED_HERO_IMAGE = BASE_DIR / "background.png"
DEFAULT_HERO_IMAGE = BASE_DIR / "new_hero.jpg"
DEMO_IMAGE_DB = DEMO_WEB_DIR / "cache/product_images.db"
DEMO_IMAGE_ROOT = DEMO_WEB_DIR.parent
APP_SETTINGS = get_settings()
LOGGER = get_logger(__name__)

MAINTENANCE_CSS = """
<style>
.mode-switch {
  position: fixed;
  top: 14px;
  right: 16px;
  z-index: 1000;
}
.mode-switch a {
  display: inline-block;
  text-decoration: none;
  background: #111827;
  color: #f9fafb;
  border: 1px solid #374151;
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
}
.mode-switch a:hover {
  background: #1f2937;
}
.maint-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.maint-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 12px;
}
.maint-label {
  color: #6b7280;
  font-size: 12px;
  margin-bottom: 4px;
}
.maint-value {
  color: #111827;
  font-size: 22px;
  font-weight: 700;
}
.maint-sub {
  color: #374151;
  font-size: 12px;
}
.maint-logs {
  background: #0f172a;
  color: #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  min-height: 280px;
  max-height: 480px;
  overflow: auto;
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
}
@media (max-width: 980px) {
  .maint-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
"""


# ───────────────────────── helpers ─────────────────────────

@st.cache_data(show_spinner=False)
def _file_to_data_uri(path: str) -> str:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except Exception:
        return ""
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }
    mime = mime_map.get(p.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _record_query_profile(query: str, mode: str, latency: float, result_count: int) -> None:
    if "query_profile" not in st.session_state:
        st.session_state.query_profile = []
    history = st.session_state.query_profile
    history.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "query": query,
            "mode": mode,
            "latency": float(latency),
            "result_count": int(result_count),
        }
    )
    max_records = 300
    if len(history) > max_records:
        del history[: len(history) - max_records]


def _is_maintenance_mode() -> bool:
    raw = st.query_params.get("maintenance", "0")
    if isinstance(raw, list):
        raw = raw[0] if raw else "0"
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _render_mode_switch(maintenance_mode: bool) -> None:
    target_mode = "0" if maintenance_mode else "1"
    label = "Back to Search" if maintenance_mode else "Open Maintenance"
    st.markdown(
        f'<div class="mode-switch"><a href="?maintenance={target_mode}">{label}</a></div>',
        unsafe_allow_html=True,
    )


def render_maintenance_dashboard() -> None:
    st.markdown(MAINTENANCE_CSS, unsafe_allow_html=True)
    st.markdown("## Maintenance Dashboard")
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5, key="maintenance_refresh")
    @st.fragment(run_every=refresh_sec)
    def _live_dashboard() -> None:
        profile = st.session_state.get("query_profile", [])
        latencies = [float(item["latency"]) for item in profile]
        query_count = len(profile)
        p95_latency = safe_percentile(latencies, 0.95)
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        last_query = profile[-1]["query"] if profile else "-"
        uptime_seconds = max(time.time() - st.session_state.get("app_start_ts", time.time()), 0.0)
        mem_mb = process_memory_mb()

        log_path = Path(APP_SETTINGS.app_log_file)
        log_tail = tail_log(log_path, max_lines=240)
        log_size_mb = (log_path.stat().st_size / (1024 * 1024)) if log_path.exists() else 0.0

        st.markdown(
            f"""
<div class="maint-grid">
  <div class="maint-card">
    <div class="maint-label">Queries (session)</div>
    <div class="maint-value">{query_count}</div>
    <div class="maint-sub">Last query: {html.escape(str(last_query))}</div>
  </div>
  <div class="maint-card">
    <div class="maint-label">Latency</div>
    <div class="maint-value">{avg_latency:.3f}s</div>
    <div class="maint-sub">p95: {p95_latency:.3f}s · max: {max_latency:.3f}s</div>
  </div>
  <div class="maint-card">
    <div class="maint-label">App process</div>
    <div class="maint-value">{mem_mb:.1f} MB</div>
    <div class="maint-sub">PID: {pyos.getpid()} · uptime: {uptime_seconds:.0f}s</div>
  </div>
  <div class="maint-card">
    <div class="maint-label">Permanent log file</div>
    <div class="maint-value">{log_size_mb:.2f} MB</div>
    <div class="maint-sub">{html.escape(str(log_path))}</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.1, 1.9])
        with col1:
            st.markdown("### Profiling Snapshot")
            if profile:
                recent = profile[-20:]
                st.dataframe(recent, use_container_width=True, height=280)
            else:
                st.info("No query profiling data yet. Run a search to populate metrics.")
        with col2:
            st.markdown("### Live Log Tail")
            st.markdown(f'<div class="maint-logs">{html.escape(log_tail)}</div>', unsafe_allow_html=True)

        st.caption(f"Auto-refresh every {refresh_sec}s · updated {datetime.utcnow().strftime('%H:%M:%S')} UTC")

    _live_dashboard()


@st.cache_resource(show_spinner="Loading models & corpus… (this can take ~1 minute)")
def load_engine(mode: str):
    if mode == "full":
        return FullSearchEngine()
    return FastSearchEngine()


@st.cache_data(show_spinner=False)
def load_image_path_map(db_path: str, web_root: str) -> dict[int, str]:
    db = Path(db_path)
    root = Path(web_root)
    web_root_dir = root / "web"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            """SELECT product_uid, local_path
               FROM product_images
               WHERE status = 'downloaded'
                 AND local_path IS NOT NULL AND local_path != ''"""
        ).fetchall()
    finally:
        conn.close()
    mapping: dict[int, str] = {}
    for uid, local_path in rows:
        try:
            uid_int = int(uid)
        except (TypeError, ValueError):
            continue
        p = Path(str(local_path))
        if not p.is_absolute():
            candidates = (
                root / p,
                web_root_dir / p,
                db.parent / p,
                BASE_DIR / p,
                BASE_DIR / "data" / p,
            )
            found = next((candidate for candidate in candidates if candidate.exists()), None)
            if found is None:
                continue
            p = found
        if not p.exists():
            continue
        mapping[uid_int] = str(p)
    return mapping


# ───────────────────────── hero banner ─────────────────────────

def render_hero_banner(hero_data_uri: str) -> None:
    if hero_data_uri:
        hero_style = (
            f"background-image: url('{hero_data_uri}');"
        )
    else:
        hero_style = "background: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);"

    st.markdown(
        f"""
<section class="hero-banner" style="{hero_style}">
  <div class="hero-content">
    <h1>Home Depot Product Search</h1>
    <p>Type your query below to find the most relevant products.</p>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────── CSS for Streamlit elements ─────────────────────────

STREAMLIT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* hide chrome */
#MainMenu, header, footer {visibility: hidden;}
div[data-testid="stDecoration"], div[data-testid="stToolbar"] {display: none;}

.stApp {
  background: #f4f6f9;
  font-family: 'Inter', Arial, sans-serif;
}
.block-container {
  max-width: 100% !important;
  padding: 0 !important;
}

/* Hero image section */
.hero-banner {
  width: 100%;
  height: 100vh;
  background-size: cover;
  background-position: center center;
  background-repeat: no-repeat;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 24px;
}
.hero-content {
  color: #111827;
  background: rgba(255, 255, 255, 0.72);
  border-radius: 12px;
  padding: 20px 28px;
}
.hero-content h1 {
  margin: 0;
  font-size: clamp(30px, 4vw, 52px);
  font-weight: 800;
  line-height: 1.1;
}
.hero-content p {
  margin: 10px 0 0;
  font-size: clamp(15px, 1.8vw, 21px);
}

/* Center the real Streamlit search bar over the hero */
.st-key-search_query {
  width: min(940px, 92vw);
  margin: calc(-1 * clamp(112px, 19vh, 200px)) auto 24px auto !important;
  position: relative;
  z-index: 100 !important;
  overflow: visible !important;
}
.st-key-search_query label { display: none !important; }
.st-key-search_query div { overflow: visible !important; }
.st-key-search_query input {
  border: 3px solid #f96302 !important;
  border-radius: 10px !important;
  padding: 16px 22px !important;
  font-size: 22px !important;
  line-height: 1.25 !important;
  font-family: 'Inter', Arial, sans-serif !important;
  box-shadow: 0 10px 28px rgba(0,0,0,.25) !important;
  min-height: 72px !important;
  height: auto !important;
  background-color: rgba(255,255,255,.96) !important;
  color: #111827 !important;
  caret-color: #111827 !important;
}
.st-key-search_query input::placeholder { color: #6b7280 !important; opacity: 1 !important; }
.st-key-search_query input:focus {
  border-color: #f96302 !important;
  box-shadow: 0 0 0 3px rgba(249,99,2,.2) !important;
  color: #111827 !important;
}
@media (max-width: 900px) {
  .hero-banner { height: 100vh; }
  .st-key-search_query {
    width: 94vw;
    margin: calc(-1 * clamp(68px, 14vh, 116px)) auto 20px auto !important;
    overflow: visible !important;
  }
  .st-key-search_query input {
    min-height: 60px !important;
    height: auto !important;
    font-size: 18px !important;
  }
}

/* product cards */
.product-card {
  border: 1px solid #e5e7eb; border-radius: 10px;
  overflow: hidden; background: #fff;
  transition: box-shadow .15s ease;
  color: #111827 !important;
}
.product-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.08); }
.product-thumb-wrap {
  width: 100%; height: 180px; background: #f3f4f6;
  display: flex; align-items: center; justify-content: center;
}
.product-thumb { width: 100%; height: 180px; object-fit: contain; }
.product-content { padding: 10px; color: #111827 !important; }
.product-rank { color: #4b5563 !important; font-size: 12px; margin-bottom: 6px; }
.product-name {
  margin: 0 0 8px; font-size: 15px; line-height: 1.3;
  min-height: 38px; color: #111827 !important;
}
.product-score { color: #065f46 !important; font-size: 13px; font-weight: 700; margin-bottom: 4px; }
.product-brand { color: #374151 !important; font-size: 12px; }
</style>
"""


# ───────────────────────── result card ─────────────────────────

def render_result_card(index: int, result: dict, image_map: dict[int, str]) -> None:
    uid = int(result["product_uid"])
    score = float(result["relevance_score"])
    title = html.escape(str(result["product_title"]))
    image_path = image_map.get(uid)
    image_data_uri = _file_to_data_uri(image_path) if image_path else ""

    if image_data_uri:
        thumb = f'<img class="product-thumb" src="{image_data_uri}" alt="product" />'
    else:
        thumb = (
            '<svg class="product-thumb" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 180">'
            '<rect width="300" height="180" fill="#f3f4f6"/>'
            '<text x="150" y="94" text-anchor="middle" fill="#9ca3af" font-size="18" '
            'font-family="sans-serif">No Image</text></svg>'
        )

    st.markdown(
        f"""
<div class="product-card">
  <div class="product-thumb-wrap">{thumb}</div>
  <div class="product-content">
    <div class="product-rank">#{index} · UID {uid}</div>
    <h4 class="product-name">{title}</h4>
    <div class="product-score">Relevance: {score:.3f} / 3.000</div>
    <div class="product-brand">Home Depot catalog item</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────── main ─────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Home Depot Search Relevance",
        page_icon="🛠️",
        layout="wide",
    )
    if "app_start_ts" not in st.session_state:
        st.session_state.app_start_ts = time.time()
    LOGGER.info("Streamlit app initialized.")

    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    maintenance_mode = _is_maintenance_mode()
    _render_mode_switch(maintenance_mode)

    if maintenance_mode:
        render_maintenance_dashboard()
    else:
        hero_source = DEFAULT_HERO_IMAGE if DEFAULT_HERO_IMAGE.exists() else UPLOADED_HERO_IMAGE
        hero_uri = _file_to_data_uri(str(hero_source)) if hero_source.exists() else ""
        render_hero_banner(hero_uri)

        # Real search input – styled via CSS to look integrated
        query = st.text_input(
            "Search products",
            placeholder="e.g., dewalt drill, ceiling fan, patio door …",
            label_visibility="collapsed",
            key="search_query",
        )

        image_map = load_image_path_map(str(DEMO_IMAGE_DB), str(DEMO_IMAGE_ROOT))

        # Default settings
        engine_mode = "fast"
        typo_mode = "aggressive"
        top_k = 8
        candidates_to_rerank = 50

        # Search execution
        if "search_lock" not in st.session_state:
            st.session_state.search_lock = threading.Lock()

        if query.strip():
            engine = load_engine(engine_mode)

            with st.spinner("Searching catalog…"):
                t0 = time.time()
                with st.session_state.search_lock:
                    results = engine.search(
                        query.strip(),
                        top_k=int(top_k),
                        candidates_to_rerank=int(candidates_to_rerank),
                        typo_mode=typo_mode,
                    )
                latency = time.time() - t0
            _record_query_profile(query=query.strip(), mode=engine_mode, latency=latency, result_count=len(results))
            LOGGER.info(
                "query=%s mode=%s typo=%s latency=%.3fs results=%d",
                query.strip(),
                engine_mode,
                typo_mode,
                latency,
                len(results),
            )

            st.success(
                f'{len(results)} results for **"{query.strip()}"** in {latency:.3f}s  ·  '
                f"mode={engine_mode}  ·  typo={typo_mode}"
            )

            cards_per_row = 4
            for row_start in range(0, len(results), cards_per_row):
                row = results[row_start : row_start + cards_per_row]
                cols = st.columns(cards_per_row)
                for offset, res in enumerate(row):
                    with cols[offset]:
                        render_result_card(row_start + offset + 1, res, image_map)


if __name__ == "__main__":
    main()
