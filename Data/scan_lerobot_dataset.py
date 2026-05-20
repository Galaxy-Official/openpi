#!/usr/bin/env python3
"""Inventory LeRobot-style task datasets for pretraining data mixing.

The script scans task folders such as:

  task_lerobot/
    data/chunk-000/file-000.parquet
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    videos/{video_key}/chunk-000/file-000.mp4
    latents/chunk-000/{feature_key}/episode_000000_0_286.pth

It writes CSV, JSON, and Markdown reports with per-task and per-modality
counts. Parquet row counts and per-column compressed sizes are read from
metadata, so large parquet files are not loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - server environment dependent.
    pq = None


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
PARQUET_SUFFIXES = {".parquet"}
LATENT_SUFFIXES = {".pth", ".pt"}
LATENT_EPISODE_RE = re.compile(
    r"episode[_-](?P<episode>\d+)[_-](?P<start>\d+)[_-](?P<end>\d+)\.(?:pth|pt)$"
)


@dataclass
class FileGroupStats:
    files: int = 0
    bytes: int = 0
    rows: Optional[int] = 0
    frames: Optional[int] = 0
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def human_bytes(value: Optional[float]) -> str:
    if value is None:
        return ""
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_rate(rate: Any) -> Optional[float]:
    if not rate:
        return None
    text = str(rate)
    if "/" in text:
        left, right = text.split("/", 1)
        denom = safe_float(right)
        if not denom:
            return None
        numer = safe_float(left)
        return None if numer is None else numer / denom
    return safe_float(text)


def read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warnings.append(f"failed to read JSON {path}: {exc}")
        return {}


def is_hidden_or_sidecar(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or path.name.startswith("._")


def iter_files(root: Path, suffixes: Optional[set[str]] = None) -> Iterable[Path]:
    if not root.exists():
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        base_path = Path(base)
        for name in files:
            path = base_path / name
            if is_hidden_or_sidecar(path.relative_to(root)):
                continue
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            yield path


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def discover_tasks(root: Path, task_glob: str, recursive: bool, tasks: Sequence[str] | None = None) -> List[Path]:
    if (root / "meta").exists() or (root / "data").exists() or root.name.endswith("_lerobot"):
        return [root.resolve()]

    if tasks:
        task_dirs = []
        missing = []
        for name in tasks:
            path = (root / name).resolve()
            if path.is_dir() and ((path / "meta").exists() or (path / "data").exists()):
                task_dirs.append(path)
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(f"Task directories not found under {root}: {missing}")
        return task_dirs

    if recursive:
        candidates = [p.parent.parent for p in root.rglob("meta/info.json")]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and p.match(task_glob)]
        if not candidates:
            candidates = [p for p in root.iterdir() if p.is_dir()]

    task_dirs = []
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        if (path / "meta").exists() or (path / "data").exists() or path.name.endswith("_lerobot"):
            task_dirs.append(path)
    return sorted(task_dirs, key=lambda p: p.name)


def modality_group(feature_key: str, dtype: str = "", source: str = "") -> str:
    key = feature_key
    if source == "latent":
        return "latent"
    if key.startswith("observation.tactiles"):
        return "tactile_video" if dtype == "video" or source == "video" else "tactile"
    if key.startswith("observation.forces"):
        return "force"
    if key.startswith("observation.images"):
        if "depth" in key:
            return "depth_image"
        return "vision_video" if dtype == "video" or source == "video" else "vision_image"
    if key.startswith("observation.state"):
        return "state"
    if key == "action":
        return "action"
    if key in {"start_pose", "end_pose"}:
        return "pose"
    if key in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
        return "index"
    return "other"


def match_feature(path_in_schema: str, feature_names: Sequence[str]) -> str:
    for feature in sorted(feature_names, key=len, reverse=True):
        if path_in_schema == feature or path_in_schema.startswith(feature + "."):
            return feature
    if ".list.element" in path_in_schema:
        return path_in_schema.split(".list.element", 1)[0]
    return path_in_schema


def parquet_metadata_direct(path: Path, feature_names: Sequence[str]) -> Dict[str, Any]:
    result = {
        "rows": None,
        "row_groups": None,
        "columns": [],
        "feature_compressed_bytes": Counter(),
        "feature_uncompressed_bytes": Counter(),
        "error": None,
    }
    size = path.stat().st_size
    if size == 0:
        result["error"] = "0-byte parquet file"
        return result
    if pq is None:
        result["error"] = "pyarrow is not installed"
        return result
    try:
        pf = pq.ParquetFile(path)
        result["rows"] = int(pf.metadata.num_rows)
        result["row_groups"] = int(pf.metadata.num_row_groups)
        column_paths = []
        for rg_index in range(pf.metadata.num_row_groups):
            row_group = pf.metadata.row_group(rg_index)
            for col_index in range(row_group.num_columns):
                col = row_group.column(col_index)
                col_path = col.path_in_schema
                feature = match_feature(col_path, feature_names)
                result["feature_compressed_bytes"][feature] += int(col.total_compressed_size or 0)
                result["feature_uncompressed_bytes"][feature] += int(col.total_uncompressed_size or 0)
                column_paths.append(col_path)
        result["columns"] = sorted(set(column_paths))
    except Exception as exc:
        result["error"] = str(exc)
    return result


def parquet_metadata(path: Path, feature_names: Sequence[str], mode: str, timeout: int) -> Dict[str, Any]:
    if mode == "direct":
        return parquet_metadata_direct(path, feature_names)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_parquet_meta_worker",
        str(path),
        json.dumps(list(feature_names)),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {
            "rows": None,
            "row_groups": None,
            "columns": [],
            "feature_compressed_bytes": {},
            "feature_uncompressed_bytes": {},
            "error": str(exc),
        }
    if proc.returncode != 0:
        return {
            "rows": None,
            "row_groups": None,
            "columns": [],
            "feature_compressed_bytes": {},
            "feature_uncompressed_bytes": {},
            "error": (proc.stderr or proc.stdout or f"worker exited {proc.returncode}").strip(),
        }
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        return {
            "rows": None,
            "row_groups": None,
            "columns": [],
            "feature_compressed_bytes": {},
            "feature_uncompressed_bytes": {},
            "error": f"failed to decode worker output: {exc}",
        }


def parquet_rows_direct(path: Path) -> Dict[str, Any]:
    if path.stat().st_size == 0:
        return {"rows": None, "row_groups": None, "error": "0-byte parquet file"}
    if pq is None:
        return {"rows": None, "row_groups": None, "error": "pyarrow is not installed"}
    try:
        pf = pq.ParquetFile(path)
        return {"rows": int(pf.metadata.num_rows), "row_groups": int(pf.metadata.num_row_groups), "error": None}
    except Exception as exc:
        return {"rows": None, "row_groups": None, "error": str(exc)}


def parquet_rows(path: Path, mode: str, timeout: int) -> Dict[str, Any]:
    if mode == "direct":
        return parquet_rows_direct(path)
    cmd = [sys.executable, str(Path(__file__).resolve()), "_parquet_rows_worker", str(path)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"rows": None, "row_groups": None, "error": str(exc)}
    if proc.returncode != 0:
        return {"rows": None, "row_groups": None, "error": (proc.stderr or proc.stdout or f"worker exited {proc.returncode}").strip()}
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        return {"rows": None, "row_groups": None, "error": f"failed to decode worker output: {exc}"}


def scan_parquet_group(
    files: List[Path],
    feature_names: Sequence[str],
    parquet_metadata_mode: str,
    parquet_timeout: int,
) -> Tuple[FileGroupStats, Dict[str, Dict[str, int]]]:
    stats = FileGroupStats(rows=0)
    by_feature: Dict[str, Dict[str, int]] = defaultdict(lambda: {"compressed_bytes": 0, "uncompressed_bytes": 0})
    for path in files:
        stats.files += 1
        stats.bytes += path.stat().st_size
        row_info = parquet_rows(path, parquet_metadata_mode, parquet_timeout)
        if row_info.get("rows") is not None:
            stats.rows = int(stats.rows or 0) + int(row_info["rows"])
        else:
            stats.errors.append(f"{path}: {row_info.get('error')}")
        if path.stat().st_size == 0:
            continue
        meta = parquet_metadata(path, feature_names, parquet_metadata_mode, parquet_timeout)
        if meta["error"]:
            stats.errors.append(f"{path}: column metadata unavailable: {meta['error']}")
        for feature, value in meta["feature_compressed_bytes"].items():
            by_feature[feature]["compressed_bytes"] += int(value)
        for feature, value in meta["feature_uncompressed_bytes"].items():
            by_feature[feature]["uncompressed_bytes"] += int(value)
    return stats, by_feature


def parquet_meta_worker_main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: _parquet_meta_worker PATH FEATURE_NAMES_JSON", file=sys.stderr)
        return 2
    path = Path(argv[0])
    feature_names = json.loads(argv[1])
    result = parquet_metadata_direct(path, feature_names)
    result["feature_compressed_bytes"] = dict(result["feature_compressed_bytes"])
    result["feature_uncompressed_bytes"] = dict(result["feature_uncompressed_bytes"])
    print(json.dumps(result, ensure_ascii=False))
    return 0


def parquet_rows_worker_main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print("usage: _parquet_rows_worker PATH", file=sys.stderr)
        return 2
    result = parquet_rows_direct(Path(argv[0]))
    print(json.dumps(result, ensure_ascii=False))
    return 0


def scan_parquet_rows_only(files: List[Path], parquet_metadata_mode: str, parquet_timeout: int) -> FileGroupStats:
    stats = FileGroupStats(rows=0)
    for path in files:
        stats.files += 1
        stats.bytes += path.stat().st_size
        row_info = parquet_rows(path, parquet_metadata_mode, parquet_timeout)
        if row_info.get("rows") is not None:
            stats.rows = int(stats.rows or 0) + int(row_info["rows"])
        else:
            stats.errors.append(f"{path}: {row_info.get('error')}")
    return stats


def video_key_from_path(path: Path, videos_dir: Path) -> str:
    rel = path.relative_to(videos_dir)
    if not rel.parts:
        return "(unknown)"
    return rel.parts[0]


def ffprobe_one(path: Path, mode: str, timeout: int) -> Dict[str, Any]:
    if mode == "none":
        return {"path": str(path), "frames": None, "duration_s": None, "error": None}
    if shutil.which("ffprobe") is None:
        return {"path": str(path), "frames": None, "duration_s": None, "error": "ffprobe not found"}

    entries = "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration"
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if mode == "count_frames":
        cmd.append("-count_frames")
        entries = "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration"
    cmd += ["-show_entries", entries, "-of", "json", str(path)]

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return {"path": str(path), "frames": None, "duration_s": None, "error": proc.stderr.strip()}
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}
        frames = safe_int(stream.get("nb_read_frames")) or safe_int(stream.get("nb_frames"))
        return {
            "path": str(path),
            "frames": frames,
            "duration_s": safe_float(stream.get("duration")),
            "fps": parse_rate(stream.get("avg_frame_rate")),
            "width": safe_int(stream.get("width")),
            "height": safe_int(stream.get("height")),
            "codec": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "error": None,
        }
    except Exception as exc:
        return {"path": str(path), "frames": None, "duration_s": None, "error": str(exc)}


def scan_videos(files: List[Path], videos_dir: Path, mode: str, jobs: int, timeout: int) -> Dict[str, FileGroupStats]:
    grouped: Dict[str, FileGroupStats] = defaultdict(lambda: FileGroupStats(frames=0))
    for path in files:
        key = video_key_from_path(path, videos_dir)
        grouped[key].files += 1
        grouped[key].bytes += path.stat().st_size

    if mode == "none" or not files:
        return grouped

    probe_results = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(ffprobe_one, path, mode, timeout): path for path in files}
        for future in as_completed(futures):
            probe_results.append(future.result())

    for item in probe_results:
        path = Path(item["path"])
        key = video_key_from_path(path, videos_dir)
        group = grouped[key]
        if item.get("error"):
            group.errors.append(f"{path}: {item['error']}")
            continue
        if item.get("frames") is not None:
            group.frames = int(group.frames or 0) + int(item["frames"])
        else:
            group.frames = None
        if item.get("duration_s") is not None:
            group.duration_s += float(item["duration_s"])
        for field_name in ("fps", "width", "height", "codec", "pix_fmt"):
            value = item.get(field_name)
            if value is not None:
                group.details.setdefault(field_name + "_values", Counter())[value] += 1
    for group in grouped.values():
        for key, value in list(group.details.items()):
            if isinstance(value, Counter):
                group.details[key] = dict(value)
    return grouped


def latent_key_from_path(path: Path, latents_dir: Path) -> str:
    rel = path.relative_to(latents_dir)
    parts = rel.parts
    if len(parts) >= 3 and parts[0].startswith("chunk-"):
        return parts[1]
    if parts:
        return parts[0]
    return "(unknown)"


def summarize_tensor_object(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    records = []
    if hasattr(obj, "shape"):
        shape = tuple(int(x) for x in obj.shape)
        records.append(
            {
                "name": prefix or "tensor",
                "type": type(obj).__name__,
                "shape": list(shape),
                "dtype": str(getattr(obj, "dtype", "")),
                "numel": int(obj.numel()) if hasattr(obj, "numel") else None,
            }
        )
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            records.extend(summarize_tensor_object(value, child_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            records.extend(summarize_tensor_object(value, child_prefix))
    else:
        records.append({"name": prefix or "object", "type": type(obj).__name__, "shape": None, "dtype": None, "numel": None})
    return records


def inspect_latent(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        import torch
    except Exception as exc:
        return [], f"torch not available: {exc}"
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        return summarize_tensor_object(obj), None
    except Exception as exc:
        return [], str(exc)


def scan_latents(files: List[Path], latents_dir: Path, inspect_mode: str) -> Dict[str, FileGroupStats]:
    grouped: Dict[str, FileGroupStats] = defaultdict(lambda: FileGroupStats(frames=0))
    for path in files:
        key = latent_key_from_path(path, latents_dir)
        group = grouped[key]
        group.files += 1
        group.bytes += path.stat().st_size
        match = LATENT_EPISODE_RE.search(path.name)
        if match:
            start = int(match.group("start"))
            end = int(match.group("end"))
            group.frames = int(group.frames or 0) + max(0, end - start)
            group.details.setdefault("episode_segments", 0)
            group.details["episode_segments"] += 1
        else:
            group.details.setdefault("unparsed_names", 0)
            group.details["unparsed_names"] += 1

    if inspect_mode == "none":
        return grouped

    for key, group in grouped.items():
        key_files = [p for p in files if latent_key_from_path(p, latents_dir) == key]
        inspect_files = key_files if inspect_mode == "all" else key_files[:1]
        tensor_records = []
        total_numel = 0
        for path in inspect_files:
            records, error = inspect_latent(path)
            if error:
                group.errors.append(f"{path}: {error}")
                continue
            for record in records:
                record["file"] = path.name
                tensor_records.append(record)
                if record.get("numel") is not None:
                    total_numel += int(record["numel"])
        if tensor_records:
            group.details["tensor_samples"] = tensor_records[:20]
            group.details["inspected_files"] = len(inspect_files)
            group.details["inspected_numel"] = total_numel
    return grouped


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def markdown_table(rows: List[Dict[str, Any]], fields: Sequence[str], max_rows: int = 40) -> str:
    if not rows:
        return "_No rows._\n"
    clipped = rows[:max_rows]
    out = []
    out.append("| " + " | ".join(fields) + " |")
    out.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in clipped:
        values = []
        for field_name in fields:
            value = row.get(field_name, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            values.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
    return "\n".join(out) + "\n"


def usable_count(record: Dict[str, Any]) -> int:
    for key in ("frames_or_rows", "expected_frames"):
        value = safe_int(record.get(key))
        if value is not None:
            return value
    return 0


def required_min(counts: Sequence[int]) -> int:
    return min(counts) if counts and all(count > 0 for count in counts) else 0


def scan_task(task_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    warnings: List[str] = []
    info = read_json(task_dir / "meta" / "info.json", warnings)
    stats_json = read_json(task_dir / "meta" / "stats.json", []) if (task_dir / "meta" / "stats.json").exists() else {}
    features = info.get("features") or {}
    feature_names = list(features.keys())

    data_files = sorted(iter_files(task_dir / "data", PARQUET_SUFFIXES))
    meta_episode_files = sorted(iter_files(task_dir / "meta" / "episodes", PARQUET_SUFFIXES))
    meta_other_files = sorted(
        p for p in iter_files(task_dir / "meta", None)
        if not p.is_relative_to(task_dir / "meta" / "episodes")
    )
    video_files = sorted(iter_files(task_dir / "videos", VIDEO_SUFFIXES))
    latent_files = sorted(iter_files(task_dir / "latents", LATENT_SUFFIXES))

    data_stats, data_feature_bytes = scan_parquet_group(
        data_files,
        feature_names,
        args.parquet_metadata_mode,
        args.parquet_timeout,
    )
    episode_stats = scan_parquet_rows_only(
        meta_episode_files,
        args.parquet_metadata_mode,
        args.parquet_timeout,
    )
    video_groups = scan_videos(video_files, task_dir / "videos", args.video_probe, args.jobs, args.ffprobe_timeout)
    latent_groups = scan_latents(latent_files, task_dir / "latents", args.latent_inspect)

    for err in data_stats.errors:
        warnings.append(f"data parquet: {err}")
    for err in episode_stats.errors:
        warnings.append(f"episode parquet: {err}")
    for key, group in video_groups.items():
        for err in group.errors:
            warnings.append(f"video {key}: {err}")
    for key, group in latent_groups.items():
        for err in group.errors:
            warnings.append(f"latent {key}: {err}")

    info_total_frames = safe_int(info.get("total_frames"))
    info_total_episodes = safe_int(info.get("total_episodes"))
    data_rows = data_stats.rows
    episode_rows = episode_stats.rows

    if info_total_frames is not None and data_rows not in (None, 0) and int(data_rows) != info_total_frames:
        warnings.append(f"info total_frames={info_total_frames} but data parquet rows={data_rows}")
    if info_total_episodes is not None and episode_rows not in (None, 0) and int(episode_rows) != info_total_episodes:
        warnings.append(f"info total_episodes={info_total_episodes} but meta episode rows={episode_rows}")

    modality_rows: List[Dict[str, Any]] = []

    for feature, meta in features.items():
        dtype = str(meta.get("dtype", ""))
        group_name = modality_group(feature, dtype=dtype, source="info")
        modality_rows.append(
            {
                "task": task_dir.name,
                "feature": feature,
                "modality_group": group_name,
                "source": "info",
                "declared_dtype": dtype,
                "shape": meta.get("shape"),
                "files": "",
                "bytes": "",
                "bytes_human": "",
                "frames_or_rows": info_total_frames if dtype in {"video", "image"} else data_rows or info_total_frames,
                "count_source": "declared_expected",
                "fps": (meta.get("info") or {}).get("video.fps") or info.get("fps"),
                "details": meta.get("info"),
            }
        )

    for feature, byte_info in sorted(data_feature_bytes.items()):
        meta = features.get(feature, {})
        dtype = str(meta.get("dtype", "parquet_column"))
        modality_rows.append(
            {
                "task": task_dir.name,
                "feature": feature,
                "modality_group": modality_group(feature, dtype=dtype, source="data"),
                "source": "data_parquet",
                "declared_dtype": dtype,
                "shape": meta.get("shape"),
                "files": data_stats.files,
                "bytes": byte_info["compressed_bytes"],
                "bytes_human": human_bytes(byte_info["compressed_bytes"]),
                "frames_or_rows": data_rows,
                "count_source": "parquet_rows",
                "fps": info.get("fps"),
                "details": {"uncompressed_bytes": byte_info["uncompressed_bytes"]},
            }
        )

    for key, group in sorted(video_groups.items()):
        meta = features.get(key, {})
        frames = group.frames
        count_source = f"ffprobe_{args.video_probe}" if frames not in (None, 0) else "expected_from_info"
        if frames in (None, 0) and info_total_frames is not None:
            frames = info_total_frames
        modality_rows.append(
            {
                "task": task_dir.name,
                "feature": key,
                "modality_group": modality_group(key, dtype="video", source="video"),
                "source": "video_files",
                "declared_dtype": meta.get("dtype", "video"),
                "shape": meta.get("shape"),
                "files": group.files,
                "bytes": group.bytes,
                "bytes_human": human_bytes(group.bytes),
                "frames_or_rows": frames,
                "count_source": count_source,
                "fps": group.details.get("fps_values") or (meta.get("info") or {}).get("video.fps") or info.get("fps"),
                "details": {
                    "duration_s": group.duration_s,
                    "probe": group.details,
                    "errors": group.errors,
                },
            }
        )

    for key, group in sorted(latent_groups.items()):
        modality_rows.append(
            {
                "task": task_dir.name,
                "feature": key,
                "modality_group": "latent",
                "source": "latent_files",
                "declared_dtype": "pth",
                "shape": None,
                "files": group.files,
                "bytes": group.bytes,
                "bytes_human": human_bytes(group.bytes),
                "frames_or_rows": group.frames,
                "count_source": "filename_span_end_minus_start" if group.frames else "file_count",
                "fps": info.get("fps"),
                "details": group.details,
            }
        )

    by_feature_source = {(row["feature"], row["source"]): row for row in modality_rows}
    def count_for(feature: str, source_preference: Sequence[str]) -> int:
        for source in source_preference:
            row = by_feature_source.get((feature, source))
            if row:
                return usable_count(row)
        return 0

    data_column_metadata_failed = any("column metadata unavailable" in err for err in data_stats.errors)

    def tabular_actual_count(feature: str) -> int:
        meta = features.get(feature, {})
        if str(meta.get("dtype", "")) in {"video", "image"}:
            return 0
        if not data_rows:
            return 0
        if feature in data_feature_bytes or data_column_metadata_failed:
            return int(data_rows)
        return 0

    wrist = count_for("observation.images.wrist", ["video_files"])
    phone = count_for("observation.images.phone", ["video_files"])
    tactile_left = count_for("observation.tactiles.left", ["video_files"])
    tactile_right = count_for("observation.tactiles.right", ["video_files"])
    force_left = tabular_actual_count("observation.forces.left")
    force_right = tabular_actual_count("observation.forces.right")
    wrist_latent = count_for("observation.images.wrist", ["latent_files"])

    pair_counts = {
        "vision_wrist": wrist,
        "vision_phone": phone,
        "tactile_left": tactile_left,
        "tactile_right": tactile_right,
        "force_left": force_left,
        "force_right": force_right,
        "latent_wrist": wrist_latent,
        "touch_force_fusion": required_min([tactile_left, tactile_right, force_left, force_right]),
        "vision_touch_force_contrast": required_min([wrist, tactile_left, tactile_right, force_left, force_right]),
        "vision_latent_contrast": required_min([wrist, wrist_latent]),
    }

    total_size = sum(path.stat().st_size for path in iter_files(task_dir, None))
    task_summary = {
        "task": task_dir.name,
        "path": str(task_dir),
        "info_total_episodes": info_total_episodes,
        "meta_episode_rows": episode_rows,
        "info_total_frames": info_total_frames,
        "data_parquet_rows": data_rows,
        "fps": info.get("fps"),
        "features_declared": len(features),
        "data_files": data_stats.files,
        "meta_episode_files": episode_stats.files,
        "video_files": len(video_files),
        "latent_files": len(latent_files),
        "data_bytes": data_stats.bytes,
        "meta_bytes": sum(p.stat().st_size for p in meta_other_files + meta_episode_files),
        "video_bytes": sum(p.stat().st_size for p in video_files),
        "latent_bytes": sum(p.stat().st_size for p in latent_files),
        "total_bytes": total_size,
        "total_human": human_bytes(total_size),
        "warnings": len(warnings),
        **pair_counts,
    }

    schema_rows = []
    for feature, meta in sorted(features.items()):
        schema_rows.append(
            {
                "task": task_dir.name,
                "feature": feature,
                "modality_group": modality_group(feature, dtype=str(meta.get("dtype", "")), source="info"),
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape"),
                "names": meta.get("names"),
                "info": meta.get("info"),
            }
        )

    return {
        "task_summary": task_summary,
        "modalities": modality_rows,
        "schema": schema_rows,
        "warnings": warnings,
        "info": info,
        "stats_keys": sorted(stats_json.keys()),
    }


def aggregate_feature_schema(schema_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in schema_rows:
        feature = row["feature"]
        item = grouped.setdefault(
            feature,
            {
                "feature": feature,
                "modality_group": row["modality_group"],
                "dtypes": set(),
                "shapes": set(),
                "tasks": set(),
                "example_info": row.get("info"),
            },
        )
        item["dtypes"].add(str(row.get("dtype")))
        item["shapes"].add(json.dumps(row.get("shape"), sort_keys=True))
        item["tasks"].add(row["task"])
    out = []
    for feature, item in sorted(grouped.items()):
        out.append(
            {
                "feature": feature,
                "modality_group": item["modality_group"],
                "dtypes": sorted(item["dtypes"]),
                "shapes": [json.loads(x) for x in sorted(item["shapes"])],
                "task_count": len(item["tasks"]),
                "tasks": sorted(item["tasks"]),
                "example_info": item["example_info"],
            }
        )
    return out


def aggregate_modality_totals(modality_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in modality_rows:
        if row["source"] == "info":
            continue
        key = (row["feature"], row["modality_group"], row["source"])
        item = grouped.setdefault(
            key,
            {
                "feature": row["feature"],
                "modality_group": row["modality_group"],
                "source": row["source"],
                "tasks": set(),
                "files": 0,
                "bytes": 0,
                "frames_or_rows": 0,
            },
        )
        item["tasks"].add(row["task"])
        item["files"] += safe_int(row.get("files")) or 0
        item["bytes"] += safe_int(row.get("bytes")) or 0
        item["frames_or_rows"] += usable_count(row)
    out = []
    for item in grouped.values():
        out.append(
            {
                **{k: v for k, v in item.items() if k != "tasks"},
                "task_count": len(item["tasks"]),
                "tasks": sorted(item["tasks"]),
                "bytes_human": human_bytes(item["bytes"]),
            }
        )
    return sorted(out, key=lambda x: (x["modality_group"], x["feature"], x["source"]))


def build_markdown_report(
    root: Path,
    task_rows: List[Dict[str, Any]],
    modality_rows: List[Dict[str, Any]],
    schema_rows: List[Dict[str, Any]],
    modality_totals: List[Dict[str, Any]],
    warnings: Dict[str, List[str]],
    args: argparse.Namespace,
) -> str:
    lines = []
    lines.append("# LeRobot Dataset Inventory\n")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Video probe mode: `{args.video_probe}`")
    lines.append(f"- Latent inspect mode: `{args.latent_inspect}`")
    lines.append("")

    lines.append("## Dataset Directory Format\n")
    lines.append("```text")
    lines.append("{root}/")
    lines.append("  {task_name}/")
    lines.append("    meta/")
    lines.append("      info.json                  # declared features, fps, total episodes/frames")
    lines.append("      stats.json                 # feature statistics when available")
    lines.append("      tasks.parquet              # task_index -> natural-language task")
    lines.append("      episodes/chunk-XXX/file-YYY.parquet")
    lines.append("                                  # episode_index, length, data/video file indices")
    lines.append("    data/chunk-XXX/file-YYY.parquet")
    lines.append("                                  # frame-level tabular data: state, action, forces, timestamps")
    lines.append("    videos/{video_key}/[chunk-XXX/]file-YYY.mp4")
    lines.append("                                  # video modalities, e.g. wrist image and tactile streams")
    lines.append("    latents/chunk-XXX/{feature_key}/episode_{episode}_{start}_{end}.pth")
    lines.append("                                  # optional latent features, usually aligned by frame span")
    lines.append("```")
    lines.append("")

    lines.append("## Task Summary\n")
    task_fields = [
        "task",
        "info_total_episodes",
        "meta_episode_rows",
        "info_total_frames",
        "data_parquet_rows",
        "vision_wrist",
        "tactile_left",
        "tactile_right",
        "force_left",
        "force_right",
        "latent_wrist",
        "touch_force_fusion",
        "vision_touch_force_contrast",
        "vision_latent_contrast",
        "total_human",
        "warnings",
    ]
    lines.append(markdown_table(task_rows, task_fields, max_rows=80))

    lines.append("## Modality Totals Across Tasks\n")
    total_fields = ["modality_group", "feature", "source", "task_count", "files", "frames_or_rows", "bytes_human"]
    lines.append(markdown_table(modality_totals, total_fields, max_rows=120))

    lines.append("## Declared Feature Schema\n")
    feature_schema = aggregate_feature_schema(schema_rows)
    schema_fields = ["feature", "modality_group", "dtypes", "shapes", "task_count"]
    lines.append(markdown_table(feature_schema, schema_fields, max_rows=120))

    lines.append("## Per-Task Modality Detail\n")
    detail_fields = ["task", "modality_group", "feature", "source", "files", "frames_or_rows", "count_source", "bytes_human"]
    non_info_rows = [row for row in modality_rows if row["source"] != "info"]
    lines.append(markdown_table(non_info_rows, detail_fields, max_rows=300))

    lines.append("## Warnings And Consistency Checks\n")
    if not any(warnings.values()):
        lines.append("_No warnings._\n")
    else:
        for task, task_warnings in warnings.items():
            if not task_warnings:
                continue
            lines.append(f"### {task}")
            for warning in task_warnings[:80]:
                lines.append(f"- {warning}")
            if len(task_warnings) > 80:
                lines.append(f"- ... {len(task_warnings) - 80} more")
            lines.append("")

    lines.append("## Output Files\n")
    lines.append("- `task_summary.csv`: one row per task, useful for pretraining sampling ratios.")
    lines.append("- `modalities_by_task.csv`: one row per task/modality/source.")
    lines.append("- `modality_totals.csv`: totals grouped by modality across all tasks.")
    lines.append("- `feature_schema.csv`: declared feature schema from `meta/info.json`.")
    lines.append("- `dataset_inventory.json`: full machine-readable inventory with warnings and probe details.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan LeRobot-style task datasets.")
    parser.add_argument("root", type=Path, help="Dataset root containing task folders.")
    parser.add_argument("--out", type=Path, default=Path("dataset_scan_report"), help="Output report directory.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Exact task directory names to scan under root. When set, --task-glob and --recursive are ignored.",
    )
    parser.add_argument("--task-glob", default="*_lerobot", help="Task directory glob for direct children.")
    parser.add_argument("--recursive", action="store_true", help="Find task dirs recursively by meta/info.json.")
    parser.add_argument(
        "--video-probe",
        choices=["none", "metadata", "count_frames"],
        default="count_frames",
        help="Use ffprobe for video metadata. count_frames is slow but most accurate.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel ffprobe jobs.")
    parser.add_argument("--ffprobe-timeout", type=int, default=300, help="Per-video ffprobe timeout in seconds.")
    parser.add_argument(
        "--parquet-metadata-mode",
        choices=["subprocess", "direct"],
        default="subprocess",
        help="Read data parquet metadata in subprocesses to isolate pyarrow native crashes.",
    )
    parser.add_argument("--parquet-timeout", type=int, default=120, help="Per-data-parquet metadata timeout in seconds.")
    parser.add_argument(
        "--latent-inspect",
        choices=["none", "sample", "all"],
        default="sample",
        help="Load no/sample/all latent .pth files with torch to record tensor shapes.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = discover_tasks(root, args.task_glob, args.recursive, args.tasks)
    if not task_dirs:
        print(f"No task directories found under {root}", file=sys.stderr)
        return 2

    all_task_rows: List[Dict[str, Any]] = []
    all_modality_rows: List[Dict[str, Any]] = []
    all_schema_rows: List[Dict[str, Any]] = []
    warnings_by_task: Dict[str, List[str]] = {}
    inventory_tasks: Dict[str, Any] = {}

    print(f"Scanning {len(task_dirs)} task directories under {root}")
    for idx, task_dir in enumerate(task_dirs, 1):
        print(f"[{idx}/{len(task_dirs)}] {task_dir.name}", flush=True)
        result = scan_task(task_dir, args)
        task_name = task_dir.name
        all_task_rows.append(result["task_summary"])
        all_modality_rows.extend(result["modalities"])
        all_schema_rows.extend(result["schema"])
        warnings_by_task[task_name] = result["warnings"]
        inventory_tasks[task_name] = result

    modality_totals = aggregate_modality_totals(all_modality_rows)
    feature_schema = aggregate_feature_schema(all_schema_rows)

    write_csv(
        out_dir / "task_summary.csv",
        all_task_rows,
        [
            "task",
            "path",
            "info_total_episodes",
            "meta_episode_rows",
            "info_total_frames",
            "data_parquet_rows",
            "fps",
            "features_declared",
            "data_files",
            "meta_episode_files",
            "video_files",
            "latent_files",
            "data_bytes",
            "meta_bytes",
            "video_bytes",
            "latent_bytes",
            "total_bytes",
            "total_human",
            "vision_wrist",
            "vision_phone",
            "tactile_left",
            "tactile_right",
            "force_left",
            "force_right",
            "latent_wrist",
            "touch_force_fusion",
            "vision_touch_force_contrast",
            "vision_latent_contrast",
            "warnings",
        ],
    )
    write_csv(
        out_dir / "modalities_by_task.csv",
        all_modality_rows,
        [
            "task",
            "modality_group",
            "feature",
            "source",
            "declared_dtype",
            "shape",
            "files",
            "frames_or_rows",
            "count_source",
            "fps",
            "bytes",
            "bytes_human",
            "details",
        ],
    )
    write_csv(
        out_dir / "modality_totals.csv",
        modality_totals,
        ["modality_group", "feature", "source", "task_count", "tasks", "files", "frames_or_rows", "bytes", "bytes_human"],
    )
    write_csv(
        out_dir / "feature_schema.csv",
        feature_schema,
        ["feature", "modality_group", "dtypes", "shapes", "task_count", "tasks", "example_info"],
    )

    inventory = {
        "root": str(root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args) | {"root": str(root), "out": str(out_dir)},
        "tasks": inventory_tasks,
        "task_summary": all_task_rows,
        "modality_totals": modality_totals,
        "feature_schema": feature_schema,
        "warnings": warnings_by_task,
    }
    with (out_dir / "dataset_inventory.json").open("w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2, default=str)

    report = build_markdown_report(root, all_task_rows, all_modality_rows, all_schema_rows, modality_totals, warnings_by_task, args)
    (out_dir / "dataset_report.md").write_text(report, encoding="utf-8")

    total_warnings = sum(len(v) for v in warnings_by_task.values())
    print(f"Done. Reports written to {out_dir}")
    print(f"Warnings: {total_warnings}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_parquet_meta_worker":
        raise SystemExit(parquet_meta_worker_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "_parquet_rows_worker":
        raise SystemExit(parquet_rows_worker_main(sys.argv[2:]))
    raise SystemExit(main())
