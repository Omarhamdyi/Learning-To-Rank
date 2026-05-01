from __future__ import annotations

import argparse
import concurrent.futures
import html
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DB_PATH = Path("web/cache/product_images.db")
IMAGE_DIR = Path("web/cache/images")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
_THREAD_LOCAL = threading.local()


def recommended_workers() -> int:
    cpu = os.cpu_count() or 8
    return min(256, max(48, cpu * 12))


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=512, pool_maxsize=512, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_catalog(data_dir: Path, max_products: int) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv", encoding="latin-1", usecols=["product_uid", "product_title"])
    test = pd.read_csv(data_dir / "test.csv", encoding="latin-1", usecols=["product_uid", "product_title"])
    merged = pd.concat([train, test], ignore_index=True)
    freq = merged["product_uid"].value_counts().rename("product_freq")
    catalog = (
        merged.drop_duplicates("product_uid")
        .merge(freq, left_on="product_uid", right_index=True, how="left")
        .sort_values("product_freq", ascending=False)
    )
    if max_products > 0:
        catalog = catalog.head(max_products)

    brand_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        data_dir / "attributes.csv", usecols=["product_uid", "name", "value"], chunksize=300_000
    ):
        brand_parts.append(chunk.loc[chunk["name"] == "MFG Brand Name", ["product_uid", "value"]])
    brand = (
        pd.concat(brand_parts, ignore_index=True)
        .drop_duplicates("product_uid")
        .rename(columns={"value": "brand"})
    )

    catalog = catalog.merge(brand, on="product_uid", how="left").fillna("")
    return catalog[["product_uid", "product_title", "brand"]].reset_index(drop=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS product_images (
            product_uid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            brand TEXT,
            image_url TEXT,
            source TEXT,
            local_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_product_images_status ON product_images(status);")
    conn.commit()


def upsert_catalog(conn: sqlite3.Connection, catalog: pd.DataFrame) -> None:
    rows = [
        (int(r.product_uid), str(r.product_title), str(r.brand), now_iso())
        for r in catalog.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO product_images (product_uid, title, brand, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(product_uid) DO UPDATE SET
            title = excluded.title,
            brand = excluded.brand,
            updated_at = excluded.updated_at;
        """,
        rows,
    )
    conn.commit()


def query_pending(
    conn: sqlite3.Connection,
    limit: int | None,
    retry_failed: bool,
) -> list[tuple[int, str, str]]:
    if retry_failed:
        cond = "status != 'downloaded' OR local_path IS NULL"
    else:
        cond = "status IN ('pending', 'no_image') OR local_path IS NULL"
    query = f"""
        SELECT product_uid, title, brand
        FROM product_images
        WHERE {cond}
        ORDER BY product_uid
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def bing_image_urls(query: str, limit: int = 12) -> list[str]:
    url = "https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
    session = get_thread_session()
    resp = session.get(url, timeout=12, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    candidates = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
    if not candidates:
        candidates = re.findall(r'"murl":"(.*?)"', resp.text)
    urls: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        image_url = html.unescape(raw).strip()
        if not image_url or not image_url.startswith(("http://", "https://")):
            continue
        if image_url in seen:
            continue
        seen.add(image_url)
        urls.append(image_url)
        if len(urls) >= limit:
            break
    return urls


def google_image_urls(query: str, limit: int = 1) -> list[str]:
    url = "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote(query)
    session = get_thread_session()
    resp = session.get(
        url,
        timeout=12,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()
    candidates = re.findall(r"https://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^\"'\s]+", resp.text)
    urls: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        image_url = html.unescape(raw).strip()
        if not image_url or not image_url.startswith(("http://", "https://")):
            continue
        if image_url in seen:
            continue
        seen.add(image_url)
        urls.append(image_url)
        if len(urls) >= limit:
            break
    return urls


def infer_extension(content_type: str, url: str) -> str:
    content_type = (content_type or "").lower().split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if content_type in mapping:
        return mapping[content_type]
    path = urllib.parse.urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download_image(uid: int, image_url: str) -> tuple[str, str]:
    attempted_urls = [image_url]
    if image_url.startswith("http://www.homedepot.com/"):
        attempted_urls.append("https://www.homedepot.com/" + image_url.removeprefix("http://www.homedepot.com/"))

    session = get_thread_session()
    last_exc: Exception | None = None
    for candidate_url in attempted_urls:
        parsed = urllib.parse.urlparse(candidate_url)
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        if parsed.scheme and parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
        header_variants = [headers]
        if "Referer" in headers:
            no_referer = dict(headers)
            no_referer.pop("Referer", None)
            header_variants.append(no_referer)
        for header_variant in header_variants:
            for timeout in (12, 20):
                try:
                    resp = session.get(
                        candidate_url,
                        timeout=timeout,
                        headers=header_variant,
                        allow_redirects=True,
                    )
                    resp.raise_for_status()
                    if not resp.content:
                        raise ValueError("Empty image content")
                    ext = infer_extension(resp.headers.get("content-type", ""), str(resp.url or candidate_url))
                    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    path = IMAGE_DIR / f"{uid}{ext}"
                    path.write_bytes(resp.content)
                    return str(path), ext
                except Exception as exc:
                    last_exc = exc

    if last_exc:
        raise last_exc
    raise ValueError("Image download failed")


def fetch_one(record: tuple[int, str, str], max_bing_candidates: int) -> dict:
    uid, title, brand = record
    title_q = str(title).strip()
    brand_q = str(brand).strip()
    if brand_q and brand_q.lower() not in title_q.lower():
        base_q = f"{title_q} {brand_q}".strip()
    else:
        base_q = title_q or brand_q
    q = f"{base_q} product photo".strip()
    fallback_q = f"{base_q} home depot product image".strip()
    compact_q = " ".join(re.sub(r"[^A-Za-z0-9 ]+", " ", base_q).split()[:8]).strip()
    title_clean = " ".join(re.sub(r"[^A-Za-z0-9 ]+", " ", title_q).split()).strip()
    short_title_6 = " ".join(title_clean.split()[:6]).strip()
    short_title_4 = " ".join(title_clean.split()[:4]).strip()
    try:
        candidate_urls: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        search_errors: list[str] = []

        def add_candidates(source: str, urls: list[str]) -> None:
            for image_url in urls:
                if image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                candidate_urls.append((source, image_url))

        google_q = title_q or base_q
        google_fallback_q = compact_q or base_q
        try:
            add_candidates("google", google_image_urls(google_q, limit=1))
        except Exception as exc:
            search_errors.append(f"google:{exc}")
        if not candidate_urls and google_fallback_q != google_q:
            try:
                add_candidates("google", google_image_urls(google_fallback_q, limit=1))
            except Exception as exc:
                search_errors.append(f"google:{exc}")

        bing_queries = [
            q,
            fallback_q,
            base_q,
            title.strip(),
            compact_q,
            short_title_6,
            short_title_4,
            brand_q,
            f"{brand_q} {short_title_4}".strip(),
        ]
        seen_queries: set[str] = set()
        for bing_q in bing_queries:
            bing_q = bing_q.strip()
            if not bing_q or bing_q in seen_queries:
                continue
            seen_queries.add(bing_q)
            try:
                add_candidates("bing", bing_image_urls(bing_q, limit=max_bing_candidates))
            except Exception as exc:
                search_errors.append(f"bing:{exc}")
            if candidate_urls:
                break

        if not candidate_urls:
            if search_errors:
                return {
                    "product_uid": uid,
                    "status": "failed",
                    "image_url": "",
                    "source": "google,bing",
                    "local_path": "",
                    "error": " | ".join(search_errors)[:300],
                }
            return {
                "product_uid": uid,
                "status": "no_image",
                "image_url": "",
                "source": "google,bing",
                "local_path": "",
                "error": "No image URL found",
            }
        download_errors: list[str] = []
        for source, image_url in candidate_urls:
            try:
                local_path, _ = download_image(uid, image_url)
                return {
                    "product_uid": uid,
                    "status": "downloaded",
                    "image_url": image_url,
                    "source": source,
                    "local_path": local_path,
                    "error": "",
                }
            except Exception as exc:
                download_errors.append(f"{source}:{exc}")
        return {
            "product_uid": uid,
            "status": "failed",
            "image_url": "",
            "source": "google,bing",
            "local_path": "",
            "error": " | ".join(download_errors)[:300] if download_errors else "Image download failed",
        }
    except Exception as exc:
        return {
            "product_uid": uid,
            "status": "failed",
            "image_url": "",
            "source": "bing",
            "local_path": "",
            "error": str(exc)[:300],
        }


def update_result(conn: sqlite3.Connection, result: dict) -> None:
    conn.execute(
        """
        UPDATE product_images
        SET image_url = ?,
            source = ?,
            local_path = ?,
            status = ?,
            error = ?,
            updated_at = ?
        WHERE product_uid = ?;
        """,
        (
            result["image_url"],
            result["source"],
            result["local_path"],
            result["status"],
            result["error"],
            now_iso(),
            int(result["product_uid"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline product image cache database.")
    parser.add_argument("--data-dir", default="data/competition")
    parser.add_argument("--max-products", type=int, default=0, help="0 means all unique products")
    parser.add_argument("--workers", type=int, default=0, help="0 means auto high-throughput worker count")
    parser.add_argument("--inflight-factor", type=int, default=8, help="Max in-flight tasks per worker")
    parser.add_argument("--max-bing-candidates", type=int, default=12, help="Bing image URL candidates to try")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress log interval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    limit = args.limit if args.limit > 0 else None
    workers = int(args.workers) if int(args.workers) > 0 else recommended_workers()
    workers = max(1, workers)
    inflight_factor = max(1, int(args.inflight_factor))
    max_inflight = max(workers, workers * inflight_factor)
    max_bing_candidates = max(1, int(args.max_bing_candidates))
    progress_every = max(1, int(args.progress_every))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print("Loading catalog...")
    catalog = load_catalog(data_dir, max_products=args.max_products)
    upsert_catalog(conn, catalog)
    pending = query_pending(conn, limit=limit, retry_failed=args.retry_failed)
    total = len(pending)
    print(f"Products indexed: {len(catalog)}")
    print(f"Products queued this run: {total}")
    print(
        f"Workers: {workers} | max_inflight={min(total, max_inflight)} "
        f"| max_bing_candidates={max_bing_candidates}"
    )

    if total == 0:
        print("No pending products to fetch.")
        return

    done = 0
    downloaded = 0
    no_image = 0
    failed = 0

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending_iter = iter(pending)
        in_flight: set[concurrent.futures.Future] = set()

        def submit_next() -> bool:
            try:
                row = next(pending_iter)
            except StopIteration:
                return False
            in_flight.add(executor.submit(fetch_one, row, max_bing_candidates))
            return True

        for _ in range(min(total, max_inflight)):
            if not submit_next():
                break

        while in_flight:
            done_futures, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done_futures:
                in_flight.remove(future)
                submit_next()
                result = future.result()
                update_result(conn, result)
                done += 1
                if result["status"] == "downloaded":
                    downloaded += 1
                elif result["status"] == "no_image":
                    no_image += 1
                else:
                    failed += 1
                if done % progress_every == 0 or done == total:
                    conn.commit()
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"Progress {done}/{total} | downloaded={downloaded} no_image={no_image} failed={failed} "
                        f"| rate={rate:.2f}/s eta={format_duration(eta)}"
                    )

    conn.commit()
    elapsed = max(time.perf_counter() - start, 1e-9)
    summary = conn.execute(
        "SELECT status, COUNT(*) FROM product_images GROUP BY status ORDER BY status;"
    ).fetchall()
    print("Final status counts:", summary)
    print(f"Run throughput: {done / elapsed:.2f} products/sec over {format_duration(elapsed)}")
    print(f"DB: {DB_PATH}")
    print(f"Images dir: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
