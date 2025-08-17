"""
Cut video clips from JSON segments and save a CSV mapping: path_to_clip, transcript.

Layout searched (recursive):
  ./naturalistic/dev/%4d/%4d/
Each leaf has paired files:
  <basename>.mp4  and  <basename>.json

JSON example:
{
  "id": "V00_S0521_I00000497_P0663",
  "metadata:transcript": [
    {"start": 15.4369, "end": 16.1970, "transcript": "Okay, okay.", "words": [...]},
    ...
  ]
}

Requires: ffmpeg & ffprobe in PATH

Usage:
  python cut_from_metadata_csv.py --root ./naturalistic/dev --out ./out --csv ./out/clips.csv --workers 4
  # Accurate but slower:
  python cut_from_metadata_csv.py --reencode
  # Dry run (prints planned cuts, no files written):
  python cut_from_metadata_csv.py --dryrun
"""

import argparse
import concurrent.futures as cf
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- time & naming helpers ----------

def hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec - math.floor(sec)) * 1000))
    s = int(math.floor(sec)) % 60
    m = (int(math.floor(sec)) // 60) % 60
    h = int(math.floor(sec)) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def coerce_time(v) -> float:
    """Accept float/int/str; interpret large ints as ms."""
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        v = v.strip()
        if re.match(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$", v):
            h, m, s = v.split(":")
            x = float(h) * 3600 + float(m) * 60 + float(s)
        else:
            x = float(v)
    else:
        raise ValueError(f"Unsupported time type: {type(v)}")
    if x > 100000:  # treat as milliseconds
        x = x / 1000.0
    return x

def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"[^a-zA-Z0-9 _-]", "", text)
    text = text.replace(" ", "_")
    return (text[:max_len].rstrip("_-")) or "clip"

# ---------- JSON parsing ----------

def load_segments(json_path: Path) -> List[Tuple[float, float, Dict]]:
    """Read segments from the 'metadata:transcript' key."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segs = data.get("metadata:transcript") or []
    if not isinstance(segs, list):
        raise ValueError(f"{json_path}: 'metadata:transcript' must be a list")

    out = []
    for i, seg in enumerate(segs, 1):
        if not isinstance(seg, dict):
            continue
        try:
            start = coerce_time(seg["start"])
            end = coerce_time(seg["end"])
        except Exception as e:
            print(f"[SKIP] {json_path.name} seg#{i}: {e}")
            continue
        if end <= start:
            print(f"[SKIP] {json_path.name} seg#{i}: end <= start ({start}, {end})")
            continue
        out.append((start, end, seg))

    out.sort(key=lambda x: x[0])
    return out

# ---------- ffprobe/ffmpeg (robust cutting) ----------

def probe_duration(src: Path) -> float:
    """Return container-reported duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(src)],
            capture_output=True, text=True, check=True
        )
        return float(r.stdout.strip())
    except Exception:
        return float("inf")

def sanitize_segment(start: float, end: float, vid_dur: float) -> Tuple[float, float]:
    """Clamp to [0, vid_dur], ensure minimal positive length."""
    eps = 1e-3
    if not math.isfinite(vid_dur):
        s = max(0.0, start)
        e = max(s + eps, end)
        return s, e
    s = max(0.0, min(start, max(0.0, vid_dur - eps)))
    e = max(s + eps, min(end, vid_dur))
    return s, e

            
def run_ffmpeg(src: Path, dst: Path, start: float, end: float, reencode: bool = False) -> Tuple[bool, str]:
    """
    Try fast stream-copy first with synthesized PTS; if that fails, fall back to re-encode
    and reset timestamps to start at 0 (fixes MOV/MP4 'out of range' errors).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, end - start)

    def _copy_cmd():
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{dur:.3f}",
            "-c", "copy",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-muxpreload", "0", "-muxdelay", "0",
            str(dst),
        ]

    def _reencode_cmd():
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
            "-vf", "setpts=PTS-STARTPTS",
            "-af", "aresample=async=1:first_pts=0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dst),
        ]

    if reencode:
        try:
            subprocess.run(_reencode_cmd(), check=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"ffmpeg re-encode failed: {e}"

    try:
        subprocess.run(_copy_cmd(), check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        try:
            subprocess.run(_reencode_cmd(), check=True)
            return True, f"stream-copy failed, fell back to re-encode ({e})"
        except subprocess.CalledProcessError as e2:
            return False, f"copy failed ({e}); re-encode failed ({e2})"

# ---------- discovery & processing ----------

def outpath(out_root: Path, rel_dir: Path, base_stem: str, idx: int, start: float, end: float, hint: Optional[str]) -> Path:
    hwin = f"{hhmmss(start).replace(':','-')}-{hhmmss(end).replace(':','-')}"
    slug = slugify(hint) if hint else "clip"
    fname = f"{base_stem}_{idx:03d}_{hwin}_{slug}.mp4"
    return out_root / rel_dir / base_stem / fname

def find_pairs(root: Path) -> List[Tuple[Path, Path, Path]]:
    """Return [(rel_dir, mp4_path, json_path)] pairing by identical basename."""
    pairs = []
    for json_file in root.rglob("*.json"):
        mp4 = json_file.with_suffix(".mp4")
        if mp4.exists():
            rel_dir = json_file.parent.relative_to(root)
            pairs.append((rel_dir, mp4, json_file))
    return sorted(pairs)

def process_pair(rel_dir: Path, mp4: Path, js: Path, out_root: Path, reencode: bool) -> Dict:
    """
    Returns:
      {
        "video": <str>,
        "json": <str>,
        "rows": [(path_to_clip, transcript), ...],
        "errors": [<str>, ...]
      }
    """
    result = {"video": str(mp4), "json": str(js), "rows": [], "errors": []}
    try:
        segs = load_segments(js)
    except Exception as e:
        result["errors"].append(f"Parse error: {e}")
        return result

    video_dur = probe_duration(mp4)
    base = mp4.stem
    for i, (s, e, seg) in enumerate(segs, 1):
        s, e = sanitize_segment(s, e, video_dur)
        hint = seg.get("transcript", "")
        dst = outpath(out_root, rel_dir, base, i, s, e, hint)
        ok, err = run_ffmpeg(mp4, dst, s, e, reencode=reencode)
        if ok:
            result["rows"].append((str(dst), hint))
            if err:
                print(f"[INFO] {dst.name}: {err}")
        else:
            result["errors"].append(f"{dst.name}: {err}")
    return result

# ---------- CSV writing ----------

def write_csv(rows: List[Tuple[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_to_clip", "transcript"])
        w.writerows(rows)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data1/open_data/seamless_interaction/naturalistic/dev"))
    ap.add_argument("--out", type=Path, default=Path("/data1/open_data/seamless_interaction/preprocessed/train"))
    ap.add_argument("--csv", type=Path, default=None, help="CSV output path (default: <out>/clips.csv)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reencode", action="store_true", help="Frame-accurate cuts (slower)")
    ap.add_argument("--dryrun", action="store_true", help="Only list planned clips")
    args = ap.parse_args()

    csv_path = args.csv or (args.out / "clips.csv")

    pairs = find_pairs(args.root)
    if not pairs:
        print(f"No (.json,.mp4) pairs found under {args.root}")
        return

    print(f"Found {len(pairs)} pairs under {args.root}")

    if args.dryrun:
        for rel_dir, mp4, js in pairs:
            try:
                segs = load_segments(js)
            except Exception as e:
                print(f"[PARSE ERR] {js}: {e}")
                continue
            print(f"\n{mp4} ({len(segs)} segments)")
            for i, (s, e, seg) in enumerate(segs, 1):
                hint = seg.get("transcript", "")
                dst = outpath(args.out, rel_dir, mp4.stem, i, s, e, hint)
                print(f"  #{i:03d} {hhmmss(s)} -> {hhmmss(e)}  => {dst} | {hint!r}")
        return

    all_rows: List[Tuple[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_pair, rel_dir, mp4, js, args.out, args.reencode)
            for rel_dir, mp4, js in pairs
        ]
        for fut in cf.as_completed(futures):
            res = fut.result()
            v = res["video"]
            if res["rows"]:
                print(f"\n[{v}] wrote {len(res['rows'])} clips")
                all_rows.extend(res["rows"])
            for err in res["errors"]:
                print(f"[ERROR] {v}: {err}")

    # Write the CSV once at the end
    write_csv(all_rows, csv_path)
    print(f"\nCSV written: {csv_path}  ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()
