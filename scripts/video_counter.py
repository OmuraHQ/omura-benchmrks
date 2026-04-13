"""Count blobs on Walrus mainnet by detected file type (magic bytes).

Streams active blobs from Blockberry (default **largest first** via ``sortBy=SIZE`` and
per-page size sort), fetches only a short prefix from the HTTP aggregator (streaming read;
does not download full blobs), then classifies with
`omura.parsers.file_detection.detect_file_type` (same as the indexer).

Usage:
  uv run python scripts/video_counter.py --max-blobs 500
  uv run python scripts/video_counter.py --json
  uv run python scripts/video_counter.py --live
  uv run python scripts/video_counter.py --blob BLOB_ID
  uv run python scripts/video_counter.py --from-sui-object 0x...
  uv run python scripts/video_counter.py --blob BLOB_ID --epoch 79
  uv run python scripts/video_counter.py --epoch 79 --workers 16 --live

Environment:
  WALRUS_AGGREGATOR_URL   default: https://walrus-mainnet-aggregator.redundex.com
  OMURA_BLOB_DISCOVERY    default: graphql (on-chain end_epoch); set blockberry for faster API
  OMURA_GRAPHQL_PAGE_SIZE objects per GraphQL page (default 50)
  OMURA_GRAPHQL_SLEEP_SEC   pause between pages (default 0.08)
  BLOCKBERRY_API_KEY      only if OMURA_BLOB_DISCOVERY=blockberry
  SUI_RPC_URL             Sui fullnode for --from-sui-object (default mainnet)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional, Set, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.parsers.file_detection import detect_file_type
from omura.utils.blockberry import (
    MANUAL_EPOCH,
    get_blob_details_by_id,
    get_current_epoch,
    is_blob_expired_for_epoch,
)
from omura.utils.blob_discovery import get_blob_discovery_source, iter_active_blob_entries

DEFAULT_AGGREGATOR = "https://walrus-mainnet-aggregator.redundex.com"
AGGREGATOR_URL = os.getenv("WALRUS_AGGREGATOR_URL", DEFAULT_AGGREGATOR).rstrip("/")

# Match file_detection: ISO BMFF / EBML / MPEG-TS benefit from ~8KiB; libmagic also uses up to 8KiB
DEFAULT_PREFIX_BYTES = 8192


def fetch_blob_prefix(
    blob_id: str, prefix_bytes: int, timeout: float
) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetch the first `prefix_bytes` from the aggregator (streaming; connection closed early).

    Returns ``(data, None)`` on success. On failure returns ``(None, kind)`` where ``kind`` is
    a short label for summaries (e.g. ``http_404``, ``timeout``) — there are no bytes to sniff.
    """
    url = f"{AGGREGATOR_URL}/v1/blobs/{blob_id}"
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, f"http_{resp.status_code}"
            return resp.raw.read(prefix_bytes), None
    except requests.Timeout:
        return None, "timeout"
    except requests.ConnectionError:
        return None, "connection"
    except requests.RequestException:
        return None, "request_error"
    except Exception:
        return None, "unknown_error"


def modality_bucket(mime_type: str, kind: str) -> str:
    """Map detect_file_type output to coarse buckets (aligned with vector store / dashboard)."""
    k = (kind or "binary").lower()
    mt = (mime_type or "").lower()
    if k == "image":
        return "image"
    if k == "video":
        return "video"
    if k == "audio":
        return "audio"
    if k in ("pdf", "text") or mt == "application/pdf" or mt.startswith("text/"):
        return "doc"
    if k == "quilt" or mt == "application/x-walrus-quilt":
        return "quilt"
    return "other"


def _classify_one(
    blob_id: str, prefix_bytes: int, timeout: float
) -> Tuple[str, Optional[str]]:
    """Returns (bucket, fetch_failure_kind or None). ``fetch_failure_kind`` is set only for ``fetch_failed``."""
    data, fetch_err = fetch_blob_prefix(blob_id, prefix_bytes, timeout)
    if data is None:
        return "fetch_failed", fetch_err or "unknown"
    if len(data) == 0:
        return "empty", None
    mime_type, _ext, kind = detect_file_type(data)
    return modality_bucket(mime_type, kind), None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Walrus active blobs and count by file type (magic bytes via detect_file_type)."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Walrus epoch for active filter (end_epoch > epoch). Default: auto via Blockberry, else MANUAL_EPOCH.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Resume Blockberry paging from this page.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="SIZE",
        help="Blockberry sortBy (default SIZE = largest blobs first; e.g. TIMESTAMP).",
    )
    parser.add_argument(
        "--order-by",
        type=str,
        default="DESC",
        help="Blockberry orderBy: ASC or DESC (default DESC).",
    )
    parser.add_argument(
        "--discovery",
        type=str,
        choices=("blockberry", "graphql", "sui_owned"),
        default=None,
        help="Blob listing backend (default: graphql = Sui GraphQL, on-chain end_epoch only).",
    )
    parser.add_argument(
        "--graphql-page-size",
        type=int,
        default=None,
        metavar="N",
        help="Sets OMURA_GRAPHQL_PAGE_SIZE for this run (GraphQL discovery only).",
    )
    parser.add_argument(
        "--graphql-sleep",
        type=float,
        default=None,
        metavar="SEC",
        help="Sets OMURA_GRAPHQL_SLEEP_SEC for this run (GraphQL discovery only).",
    )
    parser.add_argument(
        "--max-blobs",
        type=int,
        default=None,
        help="Stop after this many blobs (for sampling). Default: no limit (full scan).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("OMURA_COUNTER_WORKERS", "8")),
        help="Parallel aggregator fetches (1-64).",
    )
    parser.add_argument(
        "--prefix-bytes",
        type=int,
        default=DEFAULT_PREFIX_BYTES,
        help="Bytes to read for magic detection (256-65536; default 8192 for container sniff).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-blob HTTP timeout seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary only.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream live video count (and progress) to stderr while scanning (not with --json).",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=0.12,
        help="Minimum seconds between live status redraws (still redraws immediately on each new video).",
    )
    parser.add_argument(
        "--blob",
        type=str,
        default=None,
        metavar="BLOB_ID",
        help="Classify one blob: Blockberry expiry (GET /v1/blobs/{id}), then aggregator prefix + detect_file_type.",
    )
    parser.add_argument(
        "--from-sui-object",
        type=str,
        default=None,
        metavar="OBJECT_ID",
        help="Sui Walrus blob::Blob object id (0x...); resolves uint256 blob_id to HTTP id, then same as --blob.",
    )
    args = parser.parse_args()

    if args.blob and args.from_sui_object:
        print("Use only one of --blob or --from-sui-object", file=sys.stderr)
        return 2

    if args.blob or args.from_sui_object:
        if not (256 <= args.prefix_bytes <= 65536):
            print("--prefix-bytes must be between 256 and 65536", file=sys.stderr)
            return 2
        sui_chain_info: Optional[dict] = None
        if args.from_sui_object:
            from omura.utils.walrus_ids import aggregator_blob_id_from_sui_blob_object

            oid = args.from_sui_object.strip().strip('"').strip("'").rstrip("\\")
            bid, sui_chain_info, res_err = aggregator_blob_id_from_sui_blob_object(
                oid, timeout=min(args.timeout, 60.0)
            )
            if res_err:
                print(f"--from-sui-object: {res_err}", file=sys.stderr)
                return 2
        else:
            bid = args.blob.strip()
        current_epoch = args.epoch
        if current_epoch is None:
            current_epoch = get_current_epoch(silent=True)
        if current_epoch is None:
            current_epoch = MANUAL_EPOCH

        bb_details, bb_err = get_blob_details_by_id(bid, timeout=args.timeout)
        expired = is_blob_expired_for_epoch(bb_details, current_epoch) if bb_details else None
        blockberry_json = {
            "error": bb_err,
            "current_epoch": current_epoch,
            "start_epoch": (bb_details or {}).get("startEpoch") or (bb_details or {}).get("start_epoch"),
            "end_epoch": (bb_details or {}).get("endEpoch") or (bb_details or {}).get("end_epoch"),
            "size": (bb_details or {}).get("size") or (bb_details or {}).get("fileSize"),
            "expired": expired,
            "active": (not expired) if expired is not None else None,
        }
        if sui_chain_info:
            blockberry_json["sui_chain"] = sui_chain_info

        if not args.json:
            print(f"Aggregator: {AGGREGATOR_URL}", file=sys.stderr)
            if sui_chain_info:
                print(
                    f"Sui Blob object: {sui_chain_info.get('sui_object_id')}  "
                    f"uint256={sui_chain_info.get('blob_id_uint256')}",
                    file=sys.stderr,
                )
            print(f"Single blob (HTTP id): {bid}", file=sys.stderr)
            print(f"Blockberry expiry (vs epoch {current_epoch}; use --epoch to override):", file=sys.stderr)
            if bb_err == "no_api_key":
                print(
                    "  Skipped: BLOCKBERRY_API_KEY not set (cannot call GET /v1/blobs/{{id}}).",
                    file=sys.stderr,
                )
            elif bb_err == "not_found":
                print("  No record for this id (HTTP 404). Wrong id/network, or not in Blockberry index.", file=sys.stderr)
            elif bb_err:
                print(f"  Lookup failed: {bb_err}", file=sys.stderr)
            else:
                print(
                    f"  startEpoch={blockberry_json['start_epoch']}  "
                    f"endEpoch={blockberry_json['end_epoch']}  size={blockberry_json['size']}",
                    file=sys.stderr,
                )
                if expired is True:
                    print("  Status: expired (endEpoch <= current epoch).", file=sys.stderr)
                elif expired is False:
                    print("  Status: active (endEpoch > current epoch).", file=sys.stderr)
                else:
                    print("  Status: unknown (no endEpoch in Blockberry response).", file=sys.stderr)

        data, fetch_err = fetch_blob_prefix(bid, args.prefix_bytes, args.timeout)
        if data is None:
            kind = fetch_err or "unknown"
            if args.json:
                print(
                    json.dumps(
                        {
                            "blob_id": bid,
                            "bucket": "fetch_failed",
                            "failure_kind": kind,
                            "blockberry": blockberry_json,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"Aggregator: fetch_failed ({kind})", file=sys.stderr)
            return 1
        if len(data) == 0:
            out = {
                "blob_id": bid,
                "prefix_bytes": 0,
                "bucket": "empty",
                "mime_type": None,
                "ext": None,
                "kind": None,
                "blockberry": blockberry_json,
            }
            print(json.dumps(out, indent=2) if args.json else "empty (0 bytes prefix)")
            return 0
        mime_type, ext, kind = detect_file_type(data)
        bucket = modality_bucket(mime_type, kind)
        if args.json:
            print(
                json.dumps(
                    {
                        "blob_id": bid,
                        "prefix_bytes": len(data),
                        "bucket": bucket,
                        "mime_type": mime_type,
                        "ext": ext,
                        "kind": kind,
                        "blockberry": blockberry_json,
                    },
                    indent=2,
                )
            )
        else:
            print(f"prefix_len: {len(data)}")
            print(f"mime_type: {mime_type}")
            print(f"ext: {ext}")
            print(f"kind: {kind}")
            print(f"bucket: {bucket}")
        return 0

    if args.live and args.json:
        print("Cannot use --live with --json", file=sys.stderr)
        return 2

    if not (1 <= args.workers <= 64):
        print("--workers must be between 1 and 64", file=sys.stderr)
        return 2
    if not (256 <= args.prefix_bytes <= 65536):
        print("--prefix-bytes must be between 256 and 65536", file=sys.stderr)
        return 2
    if args.live_interval < 0.02:
        print("--live-interval must be >= 0.02", file=sys.stderr)
        return 2
    ob = args.order_by.upper()
    if ob not in ("ASC", "DESC"):
        print("--order-by must be ASC or DESC", file=sys.stderr)
        return 2

    if args.discovery:
        os.environ["OMURA_BLOB_DISCOVERY"] = args.discovery
    if args.graphql_page_size is not None:
        os.environ["OMURA_GRAPHQL_PAGE_SIZE"] = str(args.graphql_page_size)
    if args.graphql_sleep is not None:
        os.environ["OMURA_GRAPHQL_SLEEP_SEC"] = str(args.graphql_sleep)

    current_epoch = args.epoch
    if current_epoch is None:
        current_epoch = get_current_epoch()
    if current_epoch is None:
        current_epoch = MANUAL_EPOCH
        if not args.json:
            print(f"Using MANUAL_EPOCH={MANUAL_EPOCH} (could not auto-detect).", file=sys.stderr)

    counts: Counter[str] = Counter()
    fail_by_kind: Counter[str] = Counter()
    max_in_flight = max(32, args.workers * 4)

    if not args.json:
        print(f"Aggregator: {AGGREGATOR_URL}", file=sys.stderr)
        print(f"Discovery: {get_blob_discovery_source()}", file=sys.stderr)
        print(f"Active filter: end_epoch > {current_epoch}", file=sys.stderr)
        if get_blob_discovery_source() == "blockberry":
            print(
                f"Blockberry order: sortBy={args.sort_by} orderBy={ob} (per-page size sort when sortBy=SIZE)",
                file=sys.stderr,
            )
        print(f"Prefix bytes: {args.prefix_bytes}, workers: {args.workers}", file=sys.stderr)
        if args.live:
            print("Live: videos detected (updates on stderr, Ctrl+C to stop)…", file=sys.stderr)

    submitted = 0
    in_flight: Set = set()
    live_lock = threading.Lock()
    live_state = {"last_draw": 0.0}

    def _live_redraw(bucket: str) -> None:
        if not args.live or args.json:
            return
        now = time.monotonic()
        with live_lock:
            total = sum(counts.values())
            v = counts.get("video", 0)
            force = bucket == "video"
            if not force and now - live_state["last_draw"] < max(0.02, args.live_interval):
                return
            live_state["last_draw"] = now
            pct = (100.0 * v / total) if total else 0.0
            ft = counts.get("fetch_failed", 0)
            fail_suffix = ""
            if ft:
                parts = [f"{k}:{fail_by_kind[k]:,}" for k, _ in fail_by_kind.most_common(5)]
                if parts:
                    fail_suffix = "  fail_types=" + " ".join(parts)
            line = (
                f"videos={v:,}  scanned={total:,}  ({pct:.2f}% video)  "
                f"img={counts.get('image', 0):,}  aud={counts.get('audio', 0):,}  "
                f"doc={counts.get('doc', 0):,}  quilt={counts.get('quilt', 0):,}  "
                f"other={counts.get('other', 0):,}  empty={counts.get('empty', 0):,}  "
                f"fail={ft:,}{fail_suffix}"
            )
        sys.stderr.write("\r\x1b[K" + line)
        sys.stderr.flush()

    def _consume_future(fut) -> None:
        fail_kind: Optional[str] = None
        try:
            bucket, fail_kind = fut.result()
        except Exception:
            bucket = "fetch_failed"
            fail_kind = "worker_exception"
        with live_lock:
            counts[bucket] += 1
            if bucket == "fetch_failed":
                fail_by_kind[fail_kind or "unknown"] += 1
        _live_redraw(bucket)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for _page, blob_id, _meta in iter_active_blob_entries(
                current_epoch=current_epoch,
                start_page=args.start_page,
                sort_by=args.sort_by,
                order_by=ob,
            ):
                if args.max_blobs is not None and submitted >= args.max_blobs:
                    break
                fut = ex.submit(_classify_one, blob_id, args.prefix_bytes, args.timeout)
                in_flight.add(fut)
                submitted += 1

                while len(in_flight) >= max_in_flight:
                    done, not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                    for f in done:
                        _consume_future(f)
                    in_flight = not_done

            while in_flight:
                done, not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done:
                    _consume_future(f)
                in_flight = not_done
    except KeyboardInterrupt:
        if args.live and not args.json:
            sys.stderr.write("\nInterrupted.\n")
        raise

    if args.live and not args.json:
        sys.stderr.write("\n")

    total_scanned = sum(counts.values())
    payload = {
        "source": "walrus_mainnet",
        "blob_discovery": get_blob_discovery_source(),
        "epoch_filter": current_epoch,
        "aggregator": AGGREGATOR_URL,
        "blockberry_sort_by": args.sort_by,
        "blockberry_order_by": ob,
        "prefix_bytes": args.prefix_bytes,
        "total_scanned": total_scanned,
        "by_modality": {
            "image": counts.get("image", 0),
            "video": counts.get("video", 0),
            "audio": counts.get("audio", 0),
            "doc": counts.get("doc", 0),
            "quilt": counts.get("quilt", 0),
            "other": counts.get("other", 0),
        },
        "fetch_failed": counts.get("fetch_failed", 0),
        "fetch_failed_by_kind": dict(fail_by_kind.most_common()),
        "empty": counts.get("empty", 0),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n--- Walrus blob type counts (detect_file_type on prefix) ---")
        print(f"Discovery: {payload['blob_discovery']}")
        print(f"Total scanned: {total_scanned}")
        print(f"  video:   {payload['by_modality']['video']}")
        print(f"  image:   {payload['by_modality']['image']}")
        print(f"  audio:   {payload['by_modality']['audio']}")
        print(f"  doc:     {payload['by_modality']['doc']}")
        print(f"  quilt:   {payload['by_modality']['quilt']}")
        print(f"  other:   {payload['by_modality']['other']}")
        print(f"  empty:   {payload['empty']}")
        print(f"  failed:  {payload['fetch_failed']}")
        if payload["fetch_failed"] and payload["fetch_failed_by_kind"]:
            print("    failure kinds (no bytes → no MIME sniff):")
            for k, n in fail_by_kind.most_common():
                print(f"      {k}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
