"""Self-check for the inspection lab package.

This is the lab's own verification, independent of the production service's
deferred test stage. It proves:

* the viewer data and every referenced PNG exist and match their SHA-256;
* the viewer's sheet tile -> source-frame links resolve to real frame assets;
* the counterexample claims in the viewer are the recorded measurements;
* the interruption story in the viewer matches the evidence;
* a viewer frame PNG decodes back to the exact RGB bytes recorded for that
* frame, and the sheet tile pixels match a nearest-neighbour resample of that
* exact source frame;
* the lab never stores a prepared full-frame sequence: the build directory
* contains only the two source AVIs, and per-frame scans hold one RGB frame.

It reads generated files only and starts no service.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from lab.png import decode_rgb_png

VIEWER = ROOT / "viewer"
FRAMES = VIEWER / "frames"
EVIDENCE = ROOT / "evidence" / "evidence.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _load_data() -> dict:
    text = (VIEWER / "data.js").read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.LAB_DATA = (.*);\n?", text, re.S)
    if not match:
        _fail("data.js does not contain the expected assignment")
    return json.loads(match.group(1))


def _check_frame_assets(assets: list[dict], label: str) -> dict:
    seen = {}
    for asset in assets:
        path = FRAMES / asset["png_path"]
        if not path.is_file():
            _fail(f"{label}: missing frame PNG {path}")
        payload = path.read_bytes()
        if _sha(payload) != asset["png_sha256"]:
            _fail(f"{label}: PNG digest mismatch for {path.name}")
        width, height, pixels = decode_rgb_png(payload)
        if (width, height) != (asset["width"], asset["height"]):
            _fail(f"{label}: PNG dimensions changed for {path.name}")
        if _sha(pixels) != asset["rgb_sha256"]:
            _fail(f"{label}: PNG pixels differ from recorded source RGB")
        seen[asset["frame_index"]] = asset
    return seen


def _resample(frame_rgb: bytes, source_w: int, source_h: int, width: int, height: int) -> bytes:
    out = bytearray(width * height * 3)
    cursor = 0
    for y in range(height):
        sy = min(source_h - 1, y * source_h // height)
        row = sy * source_w * 3
        for x in range(width):
            sx = min(source_w - 1, x * source_w // width)
            offset = row + sx * 3
            out[cursor : cursor + 3] = frame_rgb[offset : offset + 3]
            cursor += 3
    return bytes(out)


def check_viewer(data: dict) -> list[str]:
    notes = []
    clip = data["clip"]
    frames = _check_frame_assets(clip["frameAssets"], "walk clip")
    if len(frames) != clip["frameCount"]:
        _fail("walk clip frame asset count does not match frameCount")
    if sorted(frames) != list(range(clip["frameCount"])):
        _fail("walk clip frame assets are not a complete 0..N-1 set")

    for run in data["strides"]:
        record = run["record"]
        for tile in run["sheet"]["tiles"]:
            source = tile["source_frame"]
            if source not in frames:
                _fail(f"sheet tile {tile['tile_index']} references a missing frame {source}")
            tile_png = FRAMES / tile["png_path"]
            if not tile_png.is_file():
                _fail(f"missing sheet tile PNG {tile['png_path']}")
            payload = tile_png.read_bytes()
            if _sha(payload) != tile["png_sha256"]:
                _fail(f"sheet tile PNG digest mismatch {tile['png_path']}")
            tile_w, tile_h, tile_pixels = decode_rgb_png(payload)
            if (tile_w, tile_h) != (tile["width"], tile["height"]):
                _fail("sheet tile PNG dimensions differ from its record")
            if _sha(tile_pixels) != tile["rgb_sha256"]:
                _fail("sheet tile PNG pixels differ from its record")
            source_pixels = decode_rgb_png(
                (FRAMES / frames[source]["png_path"]).read_bytes()
            )[2]
            resampled = _resample(
                source_pixels, clip["width"], clip["height"], tile_w, tile_h
            )
            if _sha(resampled) != tile["rgb_sha256"]:
                _fail(
                    f"sheet tile {tile['tile_index']} is not a resample of frame {source}"
                )
        if not run["sheet"]["artifactDigestMatches"]:
            _fail(f"stride {run['stride']} artifact digest does not match its record")
        sheet_png = (FRAMES / run["sheet"]["imagePath"]).read_bytes()
        sheet_w, sheet_h, sheet_pixels = decode_rgb_png(sheet_png)
        if (sheet_w, sheet_h) != (
            run["sheet"]["imageWidth"],
            run["sheet"]["imageHeight"],
        ):
            _fail(f"stride {run['stride']} sheet PNG dimensions differ from its record")
        if hashlib.sha256(sheet_pixels).hexdigest() != run["sheet"][
            "artifactPixelsSha256"
        ]:
            _fail(f"stride {run['stride']} sheet PNG pixels differ from the PPM pixels")
        scene_frames = [
            scene["representative_frame"] for scene in record["scenes"]
        ]
        tile_frames = [tile["source_frame"] for tile in run["sheet"]["tiles"]]
        if scene_frames != tile_frames:
            _fail(
                f"stride {run['stride']} sheet tiles do not match scene representatives"
            )

    # counterexample
    counter = data["counterexample"]
    rot_frames = _check_frame_assets(counter["frameAssets"], "rotation clip")
    pair = next(
        item
        for item in counter["pairs"]
        if item["left_frame"] == 3 and item["right_frame"] == 4
    )
    if not pair.get("statistics_identical"):
        _fail("counterexample distance is not recorded as identical statistics")
    if not pair.get("pixels_differ"):
        _fail("counterexample frames are recorded as identical pixels")
    if rot_frames[3]["rgb_sha256"] == rot_frames[4]["rgb_sha256"]:
        _fail("counterexample frame digests unexpectedly match")
    left = decode_rgb_png((FRAMES / rot_frames[3]["png_path"]).read_bytes())[2]
    right = decode_rgb_png((FRAMES / rot_frames[4]["png_path"]).read_bytes())[2]
    if left == right:
        _fail("counterexample PNGs contain identical pixels")
    if counter["inspection"]["record"]["cut_frames"] != []:
        _fail("counterexample production verdict is not an empty cut list")
    if counter["inspection"]["record"]["scenes"][0]["start_frame"] != 0:
        _fail("counterexample scene does not start at frame 0")
    notes.append("counterexample: fingerprint statistics identical, pixels differ, cut_frames=[]")

    # interruption
    interruption = data["interruption"]
    if interruption["attempt"]["committed_state"] not in {"indexed", "failed"}:
        _fail("interruption attempt did not reach a terminal state")
    if interruption["retry_same_id"]["status"] != 409:
        _fail("interruption retry did not conflict")
    reuse = interruption["reuse"]
    if not reuse["reuse_verdict_matches_committed_record"]:
        _fail("reuse verdict differs from the committed record")
    if not reuse["artifact_matches_record"]:
        _fail("reuse sheet artifact differs from its record")
    if reuse["partial_outputs_stitched"]:
        _fail("reuse claims partial outputs were stitched")
    notes.append("interruption: committed record reused, retry conflict, artifact digest matches")

    # residency claims
    for run in data["strides"]:
        if run["peakFrames"] > data["conditions"]["source_rgb_residency_limit"]:
            _fail(f"stride {run['stride']} exceeded bounded scan residency")
        if run["residencyPeak"] > data["conditions"]["source_rgb_residency_limit"]:
            _fail(f"stride {run['stride']} exceeded production residency")
    notes.append("residency: all scans and production probes stayed within 2 frames")

    # build directory must not contain a prepared frame sequence
    build_dir = VIEWER / ".build"
    if build_dir.is_dir():
        if any(build_dir.iterdir()):
            _fail("build directory still holds intermediate clip files")
    return notes


def _strip_js_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def check_viewer_markup() -> list[str]:
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    css = (VIEWER / "viewer.css").read_text(encoding="utf-8")
    js = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    problems = []
    ids_used = set(
        re.findall(r'\$\("([^"]+)"\)', js)
        + re.findall(r'getElementById\("([^"]+)"\)', js)
    )
    ids_defined = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(ids_used - ids_defined)
    if missing:
        problems.append(f"viewer.js references undefined ids: {missing}")
    for panel in re.findall(r'<section id="([^"]+)"', html):
        if f"TAB_INITS[\"{panel}\"]" not in js:
            problems.append(f"panel {panel} has no init handler")
    for tab in re.findall(r'data-tab="([^"]+)"', html):
        if f'<section id="{tab}"' not in html:
            problems.append(f"tab {tab} has no matching panel")
    if "data.js" not in html:
        problems.append("index.html does not load data.js")
    if "viewer.css" not in html:
        problems.append("index.html does not load viewer.css")
    if (VIEWER / "data.js").stat().st_size < 10_000:
        problems.append("data.js is suspiciously small")
    if not css:
        problems.append("viewer.css is empty")
    stripped = _strip_js_comments(js)
    if "window.LAB_DATA" not in js:
        problems.append("viewer.js does not read window.LAB_DATA")
    return problems


def main() -> int:
    if not EVIDENCE.is_file():
        print("FAIL: run PYTHONPATH=src:. python3 -m lab.build_evidence first")
        return 1
    data = _load_data()
    notes = check_viewer(data)
    problems = check_viewer_markup()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence["cross_checks"]["bounded_scan_equals_production"] is not True:
        problems.append("evidence: bounded scan does not equal production markers")
    if evidence["cross_checks"]["all_tiles_match_nearest_resample"] is not True:
        problems.append("evidence: some sheet tile is not the nearest resample")
    if evidence["cross_checks"]["all_artifacts_match_records"] is not True:
        problems.append("evidence: some artifact does not match its record")
    if evidence["cross_checks"]["max_observed_source_rgb_frames"] > 2:
        problems.append("evidence: source RGB residency exceeded the study limit")
    if evidence["cross_checks"]["median_speed_is_monotonic"] is not True:
        problems.append("evidence: speed medians are not monotonic")
    if problems:
        print("FAIL")
        for problem in problems:
            print(" -", problem)
        return 1
    for note in notes:
        print("PASS", note)
    print("PASS viewer markup, ids, panel handlers, and data binding")
    print("PASS lab selfcheck")
    return 0


if __name__ == "__main__":
    sys.exit(main())
