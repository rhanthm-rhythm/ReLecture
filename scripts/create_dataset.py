from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


CATALOG_BASE_URL = "https://app.underline.io/api/v1/thin_lectures"
TRANSCRIPTS_BASE_URL = "https://app.underline.io/api/v1/transcripts"
DEFAULT_LINKS_FILE = Path("data") / "video_links.txt"
DEFAULT_OUTPUT_DIR = Path("data") / "dataset"
CATALOG_CACHE_FILENAME = ".underline_catalog_cache.json"
DATASET_INDEX_FILENAME = "dataset_index.json"
API_HEADERS = {"Accept": "application/vnd.api+json"}
UNDERLINE_URL_RE = re.compile(
    r"^https://assets\.underline\.io/video/(?P<video_id>\d+)/file/abr/(?P<asset_hash>[0-9a-f]+)\.m3u8$"
)


@dataclass(frozen=True)
class UnderlineAsset:
    source_url: str
    video_id: str
    asset_hash: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an Underline-backed video dataset with transcripts.")
    parser.add_argument("--links-file", default=str(DEFAULT_LINKS_FILE), help="Text file containing one Underline HLS URL per line.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where dataset items will be written.")
    parser.add_argument("--catalog-page-size", type=int, default=500, help="Page size used when crawling the public Underline catalog.")
    parser.add_argument("--limit", type=int, help="Only process the first N links from the input file.")
    parser.add_argument("--skip-video", action="store_true", help="Resolve metadata and transcripts without downloading videos.")
    parser.add_argument("--skip-transcripts", action="store_true", help="Download videos only.")
    parser.add_argument("--download-pdf", action="store_true", help="Also download transcript PDFs when Underline exposes them.")
    parser.add_argument("--force-refresh-catalog", action="store_true", help="Ignore the local catalog cache and rebuild it.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args(argv)


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or fallback


def parse_underline_asset_url(url: str) -> UnderlineAsset:
    normalized = url.strip()
    match = UNDERLINE_URL_RE.match(normalized)
    if not match:
        raise ValueError(f"Unsupported Underline asset URL: {url}")
    return UnderlineAsset(
        source_url=normalized,
        video_id=match.group("video_id"),
        asset_hash=match.group("asset_hash"),
    )


def read_links(path: Path, limit: int | None = None) -> list[UnderlineAsset]:
    assets: list[UnderlineAsset] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        asset = parse_underline_asset_url(stripped)
        if asset.source_url in seen:
            continue
        seen.add(asset.source_url)
        assets.append(asset)
        if limit is not None and len(assets) >= limit:
            break
    return assets


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError("ffmpeg is required to download HLS videos but was not found on PATH.")


def load_catalog_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {"playlist_map": {}, "fetched_pages": 0}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_catalog_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def fetch_json(session: requests.Session, url: str, *, params: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    response = session.get(url, headers=API_HEADERS, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def crawl_catalog_until_resolved(
    session: requests.Session,
    assets: list[UnderlineAsset],
    cache_path: Path,
    *,
    page_size: int,
    timeout: float,
    force_refresh: bool,
) -> dict[str, dict[str, Any]]:
    unresolved = {asset.source_url for asset in assets}
    cache = {"playlist_map": {}, "fetched_pages": 0} if force_refresh else load_catalog_cache(cache_path)
    playlist_map = dict(cache.get("playlist_map", {}))
    for playlist_url in list(unresolved):
        if playlist_url in playlist_map:
            unresolved.discard(playlist_url)
    if not unresolved:
        return playlist_map

    params = {
        "filter[library]": "true",
        "filter[scope]": "published",
        "page[size]": str(page_size),
    }
    page_number = int(cache.get("fetched_pages", 0)) + 1
    total_pages: int | None = None

    while unresolved and (total_pages is None or page_number <= total_pages):
        params["page[number]"] = str(page_number)
        payload = fetch_json(session, CATALOG_BASE_URL, params=params, timeout=timeout)
        total_pages = int(payload.get("meta", {}).get("total_pages", 0) or 0)
        records = payload.get("data", [])
        if not records:
            break

        for record in records:
            attributes = record.get("attributes", {})
            playlist_url = attributes.get("playlist")
            if not isinstance(playlist_url, str) or not playlist_url.startswith("https://assets.underline.io/video/"):
                continue
            playlist_map[playlist_url] = {
                "lecture_id": record.get("id"),
                "title": attributes.get("title"),
                "slug": attributes.get("slug"),
                "event_id": attributes.get("event_id"),
                "published": attributes.get("published"),
                "playlist": playlist_url,
                "slideshow_url": attributes.get("slideshow_url"),
                "underline_doi": attributes.get("underline_doi"),
                "video_doi": attributes.get("video_doi"),
            }
            unresolved.discard(playlist_url)

        cache["playlist_map"] = playlist_map
        cache["fetched_pages"] = page_number
        save_catalog_cache(cache_path, cache)
        print(f"[catalog] crawled page {page_number}/{total_pages or '?'}; unresolved={len(unresolved)}")
        page_number += 1

    return playlist_map


def fetch_transcripts_for_lecture(
    session: requests.Session,
    lecture_id: str,
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    payload = fetch_json(
        session,
        TRANSCRIPTS_BASE_URL,
        params={
            "filter[lecture_id]": lecture_id,
            "filter[scope]": "with_subtitles",
            "include": "language",
            "page[size]": "100",
        },
        timeout=timeout,
    )
    included_by_id = {
        item["id"]: item.get("attributes", {})
        for item in payload.get("included", [])
        if item.get("type") == "transcript_languages"
    }
    transcripts: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        language_rel = item.get("relationships", {}).get("language", {}).get("data", {})
        language_attrs = included_by_id.get(language_rel.get("id"), {})
        transcripts.append(
            {
                "id": item.get("id"),
                "subtitle_url": attrs.get("subtitle_url") or attrs.get("subtitleUrl"),
                "pdf_url": attrs.get("pdf_url") or attrs.get("pdfUrl"),
                "auto_generated": attrs.get("auto_generated") if "auto_generated" in attrs else attrs.get("autoGenerated"),
                "language_name": language_attrs.get("name"),
                "language_locale": language_attrs.get("locale"),
            }
        )
    return transcripts


def infer_extension_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix or fallback


def item_directory_name(asset: UnderlineAsset, lecture_meta: dict[str, Any] | None) -> str:
    if lecture_meta and lecture_meta.get("lecture_id") and lecture_meta.get("slug"):
        slug = slugify(str(lecture_meta["slug"]), fallback=asset.asset_hash[:12])
        return f"{lecture_meta['lecture_id']}_{slug}"
    return asset.asset_hash


def download_file(session: requests.Session, url: str, destination: Path, *, timeout: float, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_video(asset: UnderlineAsset, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y" if overwrite else "-n",
        "-i",
        asset.source_url,
        "-c",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        str(destination),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed for {asset.source_url}: {stderr}")


def choose_transcript(transcripts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not transcripts:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        locale = str(item.get("language_locale") or "")
        name = str(item.get("language_name") or "")
        return (
            0 if locale == "en" else 1,
            0 if "english" in name.lower() else 1,
            name.lower(),
        )

    return sorted(transcripts, key=sort_key)[0]


def build_summary_record(
    asset: UnderlineAsset,
    lecture_meta: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    lecture_url = None
    if lecture_meta and lecture_meta.get("lecture_id") and lecture_meta.get("slug"):
        lecture_url = f"https://underline.io/lecture/{lecture_meta['lecture_id']}-{lecture_meta['slug']}"
    return {
        "source_url": asset.source_url,
        "video_id": asset.video_id,
        "asset_hash": asset.asset_hash,
        "lecture_id": lecture_meta.get("lecture_id") if lecture_meta else None,
        "lecture_slug": lecture_meta.get("slug") if lecture_meta else None,
        "title": lecture_meta.get("title") if lecture_meta else None,
        "lecture_url": lecture_url,
        "output_dir": str(output_dir),
    }


def process_asset(
    session: requests.Session,
    asset: UnderlineAsset,
    lecture_meta: dict[str, Any] | None,
    output_root: Path,
    *,
    timeout: float,
    overwrite: bool,
    download_pdf: bool,
    skip_video: bool,
    skip_transcripts: bool,
) -> dict[str, Any]:
    item_dir = output_root / item_directory_name(asset, lecture_meta)
    item_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_record(asset, lecture_meta, item_dir)
    summary["status"] = "pending"

    metadata_path = item_dir / "metadata.json"
    video_path = item_dir / "video.mp4"
    summary["files"] = {
        "metadata": str(metadata_path),
        "video": str(video_path),
        "transcript_vtt": None,
        "transcript_pdf": None,
    }

    if not skip_video:
        download_video(asset, video_path, overwrite=overwrite)

    transcripts: list[dict[str, Any]] = []
    selected_transcript: dict[str, Any] | None = None
    if lecture_meta and lecture_meta.get("lecture_id") and not skip_transcripts:
        transcripts = fetch_transcripts_for_lecture(session, str(lecture_meta["lecture_id"]), timeout=timeout)
        selected_transcript = choose_transcript(transcripts)
        if selected_transcript and selected_transcript.get("subtitle_url"):
            transcript_ext = infer_extension_from_url(str(selected_transcript["subtitle_url"]), ".vtt")
            transcript_path = item_dir / f"transcript{transcript_ext}"
            download_file(
                session,
                str(selected_transcript["subtitle_url"]),
                transcript_path,
                timeout=timeout,
                overwrite=overwrite,
            )
            summary["files"]["transcript_vtt"] = str(transcript_path)
        if download_pdf and selected_transcript and selected_transcript.get("pdf_url"):
            pdf_path = item_dir / "transcript.pdf"
            download_file(
                session,
                str(selected_transcript["pdf_url"]),
                pdf_path,
                timeout=timeout,
                overwrite=overwrite,
            )
            summary["files"]["transcript_pdf"] = str(pdf_path)

    summary["transcripts"] = transcripts
    summary["selected_transcript"] = selected_transcript
    summary["status"] = "ok"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    links_file = Path(args.links_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = read_links(links_file, limit=args.limit)
    if not assets:
        print("No valid Underline links were found.", file=sys.stderr)
        return 1

    if not args.skip_video:
        check_ffmpeg_available()

    session = requests.Session()
    session.headers.update({"User-Agent": "ReLecture dataset bootstrap/1.0"})

    cache_path = output_dir / CATALOG_CACHE_FILENAME
    playlist_map = crawl_catalog_until_resolved(
        session,
        assets,
        cache_path,
        page_size=args.catalog_page_size,
        timeout=args.timeout,
        force_refresh=args.force_refresh_catalog,
    )

    summaries: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        lecture_meta = playlist_map.get(asset.source_url)
        label = lecture_meta.get("title") if lecture_meta else asset.asset_hash
        print(f"[item {index}/{len(assets)}] {label}")
        try:
            summary = process_asset(
                session,
                asset,
                lecture_meta,
                output_dir,
                timeout=args.timeout,
                overwrite=args.overwrite,
                download_pdf=args.download_pdf,
                skip_video=args.skip_video,
                skip_transcripts=args.skip_transcripts,
            )
        except Exception as exc:  # noqa: BLE001
            summary = build_summary_record(asset, lecture_meta, output_dir / item_directory_name(asset, lecture_meta))
            summary["status"] = "error"
            summary["error"] = str(exc)
        summaries.append(summary)
        time.sleep(0.05)

    dataset_index_path = output_dir / DATASET_INDEX_FILENAME
    dataset_index_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    ok_count = sum(1 for item in summaries if item.get("status") == "ok")
    error_count = len(summaries) - ok_count
    missing_catalog_count = sum(1 for item in summaries if item.get("lecture_id") is None)
    transcript_count = sum(1 for item in summaries if item.get("files", {}).get("transcript_vtt"))
    print(
        f"Finished. ok={ok_count} error={error_count} unresolved_catalog={missing_catalog_count} "
        f"with_transcript={transcript_count} index={dataset_index_path}"
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
