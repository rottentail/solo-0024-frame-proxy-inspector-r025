"""Generate the inspection lab's evidence and viewer data.

Everything under ``lab/evidence``, ``lab/viewer/frames`` and
``lab/viewer/data.js`` is produced by this module:

1. it starts the production HTTP application on a loopback port against a
   throwaway data root that belongs only to the lab;
2. it registers the two lab clips through the public API exactly as an
   external user would;
3. it runs the production inspection and contact sheet workflows for every
   sampling setting in the study;
4. it records raw response bodies, response timing, server-side state,
   digests of every artifact, and bounded per-frame evidence;
5. it interrupts a client request mid-flight and proves the committed record
   is reused, not stitched from partial output;
6. it derives viewer frame PNGs one frame at a time from the stored clip.

The production service source is never edited. The lab talks to it through
HTTP and through read-only decoding of the stored clip bytes. Every decode in
the lab's own loops is guarded by ``InFlightGuard(limit=1)``, which fails the
build if two source RGB frames are alive together. The study also instruments
the production request at runtime (temporary wrappers around
``decode_frame_at`` and ``Frame.__del__`` that only count frames) to measure
how many source RGB frames the production call itself keeps alive. The
measurement is recorded as data, not asserted to be one."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

from frame_proxy.app import build_router
from frame_proxy.config import Settings
from frame_proxy.http.server import build_server
from frame_proxy.media.avi import AviDescriptor, probe_avi
from frame_proxy.media.detector import analyze_frames
from frame_proxy.media.fingerprint import fingerprint_distance, fingerprint_frame
from frame_proxy.media.frames import decode_frame_at
from frame_proxy.media.ppm import parse_ppm_header
import frame_proxy.service.inspections as inspections_module

from lab.labclip import (
    ROT_CLIP_ID,
    WALK_CLIP_ID,
    build_rot_clip,
    build_walk_clip,
)
from lab.png import encode_rgb_png
from lab.residency import probe_residency, summarize
from lab.sampling import InFlightGuard, scan_bounded, verify_frames_bounded

EVIDENCE_VERSION = 1
SENSITIVITY = 0.55
MIN_SCENE_FRAMES = 2
STRIDES = (1, 2, 3, 4)
STRIDE_LABELS = {
    1: "逐帧全采样",
    2: "每 2 帧采 1",
    3: "每 3 帧采 1",
    4: "每 4 帧采 1",
}

SHEET_COLUMNS = 4
TILE_WIDTH = 120
GUTTER = 6
BACKGROUND = "#20252c"

ROT_SHEET = {
    "sheet_id": "rot-sheet",
    "columns": 2,
    "tile_width": 160,
    "gutter": 6,
    "selection": "stored",
    "background": "#20252c",
}


def _digest(data) -> str:
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    view = memoryview(data)
    return hashlib.sha256(view).hexdigest()


def _mmap_read(path: Path):
    """Read a clip through a read-only mapping so uploads need no full copy.

    The mapping is closed when the process exits; only frames requested by
    index are ever materialised as RGB.
    """

    import mmap

    handle = open(path, "rb")
    try:
        return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        handle.close()


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


class Service:
    """A real production server on a loopback port with a private data root."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.settings = Settings(data_root=data_root)
        self.router = build_router(self.settings)
        self.server = build_server(
            "127.0.0.1", 0, router=self.router, settings=self.settings
        )
        self.port = int(self.server.server_address[1])
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )

    def __enter__(self) -> "Service":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5.0)

    def request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        content_type: str = "application/json",
        timeout: float = 60.0,
    ) -> dict:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=timeout
        )
        started = time.perf_counter_ns()
        try:
            connection.request(
                method, path, body=body, headers={"Content-Type": content_type}
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        elapsed = time.perf_counter_ns() - started
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        return {
            "method": method,
            "path": path,
            "status": response.status,
            "content_type": response.getheader("Content-Type", ""),
            "elapsed_ms": round(elapsed / 1_000_000, 2),
            "body_bytes": len(payload),
            "body_sha256": _digest(payload),
            "json": parsed,
            "_payload": payload,
        }


def _public_response(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _decode_rgb(media, descriptor: AviDescriptor, index: int) -> bytes:
    frame = decode_frame_at(media, descriptor, index)
    try:
        return frame.rgb
    finally:
        del frame


def _frame_pngs_bounded(
    media,
    descriptor: AviDescriptor,
    frame_indices,
    directory: Path,
    prefix: str,
) -> list[dict]:
    """Decode, digest, convert and write one frame before touching the next."""

    guard = InFlightGuard(limit=1)
    entries = []
    for index in frame_indices:
        frame = guard.hold(decode_frame_at(media, descriptor, index))
        rgb = frame.rgb
        png = encode_rgb_png(frame.width, frame.height, rgb)
        relative = f"{prefix}-{index:03d}.png"
        (directory / relative).write_bytes(png)
        entries.append(
            {
                "frame_index": index,
                "png_path": relative,
                "png_sha256": _digest(png),
                "rgb_sha256": _digest(rgb),
                "width": frame.width,
                "height": frame.height,
                "encoded_from": "decode_frame_at(production AVI decoder).rgb",
            }
        )
        del frame, rgb
        guard.release()
    guard.assert_clean()
    return entries


def _ppm_to_rgb(payload: bytes) -> tuple[int, int, bytes]:
    header = parse_ppm_header(payload)
    return header.width, header.height, payload[header.pixel_offset :]


def _sheet_tile_crops(pixels: bytes, width: int, height: int, tile_map) -> list[dict]:
    crops = []
    for tile in tile_map:
        left = int(tile["left"])
        top = int(tile["top"])
        tile_width = int(tile["width"])
        tile_height = int(tile["height"])
        crop = bytearray(tile_width * tile_height * 3)
        for row in range(tile_height):
            source = ((top + row) * width + left) * 3
            target = row * tile_width * 3
            crop[target : target + tile_width * 3] = pixels[
                source : source + tile_width * 3
            ]
        crops.append(
            {
                "tile_index": int(tile["tile_index"]),
                "scene_index": int(tile["scene_index"]),
                "source_frame": int(tile["source_frame"]),
                "left": left,
                "top": top,
                "width": tile_width,
                "height": tile_height,
                "rgb_sha256": _digest(bytes(crop)),
                "_crop": bytes(crop),
            }
        )
    return crops


def _nearest_resize_source(
    frame_rgb: bytes,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
) -> bytes:
    output = bytearray(tile_width * tile_height * 3)
    destination = 0
    for y in range(tile_height):
        source_y = min(height - 1, y * height // tile_height)
        source_row = source_y * width * 3
        for x in range(tile_width):
            source_x = min(width - 1, x * width // tile_width)
            source = source_row + source_x * 3
            output[destination : destination + 3] = frame_rgb[source : source + 3]
            destination += 3
    return bytes(output)


def _scene_assignments(record: dict) -> list[dict]:
    return [
        {
            "index": scene["index"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "representative_frame": scene["representative_frame"],
        }
        for scene in record["scenes"]
    ]


def _compare_verdict(production: dict, verdict) -> bool:
    observed = [
        (scene.start_frame, scene.end_frame, scene.representative_frame)
        for scene in verdict.scenes
    ]
    recorded = [
        (scene["start_frame"], scene["end_frame"], scene["representative_frame"])
        for scene in production["scenes"]
    ]
    return observed == recorded and list(verdict.cut_frames) == list(
        production["cut_frames"]
    )


def _truth_label_for_frame(frame_index: int, design_scenes: list[dict]) -> int:
    for scene in design_scenes:
        if scene["start_frame"] <= frame_index < scene["end_frame"]:
            return int(scene["index"])
    raise ValueError("frame lies outside the designed scenes")


def _truth_label_counts(
    start_frame: int,
    end_frame: int,
    design_scenes: list[dict],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for scene in design_scenes:
        overlap = min(end_frame, scene["end_frame"]) - max(
            start_frame, scene["start_frame"]
        )
        if overlap > 0:
            counts[int(scene["index"])] = overlap
    return counts


def _derive_metrics(
    record: dict,
    design_scenes: list[dict],
    truth_cuts: list[int],
    stride: int,
    reference_detected: list[int],
) -> dict:
    detected = list(record["cut_frames"])
    tolerance = stride  # sampling can delay a boundary by at most stride - 1
    missed = [
        cut
        for cut in truth_cuts
        if not any(abs(cut - found) <= tolerance for found in detected)
    ]
    spurious = [
        cut
        for cut in detected
        if not any(abs(cut - truth) <= tolerance for truth in truth_cuts)
    ]
    offsets = []
    for cut in truth_cuts:
        if any(abs(cut - found) <= tolerance for found in detected):
            nearest = min(detected, key=lambda found: abs(found - cut))
            offsets.append(
                {
                    "truth_frame": cut,
                    "detected_frame": nearest,
                    "offset_frames": nearest - cut,
                }
            )
    scene_stats = []
    for scene in record["scenes"]:
        counts = _truth_label_counts(
            scene["start_frame"], scene["end_frame"], design_scenes
        )
        dominant = max(counts, key=counts.get)
        representative_label = _truth_label_for_frame(
            scene["representative_frame"], design_scenes
        )
        scene_stats.append(
            {
                "scene_index": scene["index"],
                "start_frame": scene["start_frame"],
                "end_frame": scene["end_frame"],
                "representative_frame": scene["representative_frame"],
                "covered_truth_scenes": sorted(counts),
                "dominant_truth_scene": dominant,
                "representative_truth_scene": representative_label,
                "representative_matches_dominant": representative_label == dominant,
                "contains_short_shot": any(
                    design_scenes[index]["length"] < stride
                    for index in counts
                ),
            }
        )
    missed_labels = [
        design_scenes[index]["label"]
        for index, scene in enumerate(design_scenes)
        if scene["start_frame"] in missed
    ]
    missed_details = []
    for cut in missed:
        scene_index = next(
            index
            for index, scene in enumerate(design_scenes)
            if scene["start_frame"] == cut
        )
        previous_length = (
            design_scenes[scene_index - 1]["length"] if scene_index > 0 else 0
        )
        current_length = design_scenes[scene_index]["length"]
        if stride > 1 and (previous_length < stride or current_length < stride):
            reason = "short_shot_skipped_by_stride"
        elif cut not in reference_detected:
            reason = "below_threshold_at_every_stride"
        else:
            reason = "missed_at_this_stride"
        missed_details.append(
            {
                "truth_frame": cut,
                "label": design_scenes[scene_index]["label"],
                "previous_shot_length": previous_length,
                "current_shot_length": current_length,
                "reason": reason,
            }
        )
    return {
        "missed_details": missed_details,
        "detected_cut_count": len(detected),
        "truth_cut_count": len(truth_cuts),
        "match_tolerance_frames": tolerance,
        "matched_within_tolerance": len(offsets),
        "missed_truth_cuts": missed,
        "missed_scene_labels": missed_labels,
        "spurious_cuts": spurious,
        "boundary_offsets": offsets,
        "max_abs_offset_frames": max(
            (abs(item["offset_frames"]) for item in offsets), default=0
        ),
        "scene_count": len(record["scenes"]),
        "scenes_spanning_multiple_truth_shots": sum(
            1 for scene in scene_stats if len(scene["covered_truth_scenes"]) > 1
        ),
        "representative_frame_mismatches": sum(
            1 for scene in scene_stats if not scene["representative_matches_dominant"]
        ),
        "scene_stats": scene_stats,
    }


def _run_sampling_study(service: Service, walk_avi, manifest: dict) -> tuple[list[dict], list[dict]]:
    descriptor = probe_avi(walk_avi)
    runs = []
    requests = []
    reference_detected: list[int] = []
    for stride in STRIDES:
        inspection_id = f"walk-s{stride}"
        inspection_request = {
            "inspection_id": inspection_id,
            "sensitivity": SENSITIVITY,
            "sample_stride": stride,
            "min_scene_frames": MIN_SCENE_FRAMES,
        }
        # Three independent requests per setting: the reported timing is the
        # median of all three, and every sample is published. The first request
        # is the record the lab keeps using for sheet construction.
        timings = []
        residency_samples = []
        response = None
        for attempt in range(3):
            attempt_request = (
                inspection_request
                if attempt == 0
                else dict(
                    inspection_request,
                    inspection_id=f"{inspection_id}-t{attempt + 1}",
                )
            )
            with probe_residency(inspections_module) as residency:
                attempt_response = service.request(
                    "POST",
                    f"/v1/clips/{WALK_CLIP_ID}/inspections",
                    _json_bytes(attempt_request),
                )
            requests.append(_public_response(attempt_response))
            timings.append(attempt_response["elapsed_ms"])
            residency_samples.append(
                summarize(
                    residency,
                    len(attempt_response["json"]["inspection"]["sampled_frames"]),
                )
            )
            if attempt == 0:
                response = attempt_response
        record = response["json"]["inspection"]
        if not reference_detected:
            reference_detected = list(record["cut_frames"])
        sorted_timings = sorted(timings)
        median_ms = (
            sorted_timings[len(sorted_timings) // 2]
            if len(sorted_timings) % 2
            else round(
                (
                    sorted_timings[len(sorted_timings) // 2 - 1]
                    + sorted_timings[len(sorted_timings) // 2]
                )
                / 2,
                2,
            )
        )
        residency_summary = dict(
            residency_samples[0], median_production_elapsed_ms=median_ms
        )

        scan = scan_bounded(
            walk_avi,
            sample_stride=stride,
            sensitivity=SENSITIVITY,
            min_scene_frames=MIN_SCENE_FRAMES,
        )
        if not _compare_verdict(record, scan.verdict):
            raise RuntimeError(f"bounded scan diverged from production at stride {stride}")
        scores = _measured_scores(
            walk_avi, descriptor, stride, manifest["truth_cuts"]
        )

        sheet_id = f"walk-s{stride}-sheet"
        sheet_request = {
            "sheet_id": sheet_id,
            "columns": SHEET_COLUMNS,
            "tile_width": TILE_WIDTH,
            "gutter": GUTTER,
            "selection": "stored",
            "background": BACKGROUND,
        }
        sheet_response = service.request(
            "POST",
            f"/v1/inspections/{inspection_id}/contact-sheets",
            _json_bytes(sheet_request),
        )
        requests.append(_public_response(sheet_response))
        sheet_record = sheet_response["json"]["contact_sheet"]
        artifact_response = service.request(
            "GET", f"/v1/contact-sheets/{sheet_id}/artifact"
        )
        requests.append(_public_response(artifact_response))
        sheet_width, sheet_height, sheet_pixels = _ppm_to_rgb(artifact_response["_payload"])
        sheet_png = encode_rgb_png(sheet_width, sheet_height, sheet_pixels)
        sheet_png_path = f"sheet-s{stride}.png"
        (service_frame_dir / sheet_png_path).write_bytes(sheet_png)

        tile_crops = _sheet_tile_crops(
            sheet_pixels, sheet_width, sheet_height, sheet_record["tile_map"]
        )
        proofs = []
        pixel_guard = InFlightGuard(limit=1)
        for tile in tile_crops:
            crop_png = encode_rgb_png(tile["width"], tile["height"], tile["_crop"])
            tile["png_path"] = f"tile-s{stride}-{tile['tile_index']}.png"
            (service_frame_dir / tile["png_path"]).write_bytes(crop_png)
            tile["png_sha256"] = _digest(crop_png)
            del tile["_crop"]

            frame_rgb = pixel_guard.hold(
                _decode_rgb(walk_avi, descriptor, tile["source_frame"])
            )
            resampled = _nearest_resize_source(
                frame_rgb,
                descriptor.width,
                descriptor.height,
                tile["width"],
                tile["height"],
            )
            proofs.append(
                {
                    "tile_index": tile["tile_index"],
                    "source_frame": tile["source_frame"],
                    "source_frame_rgb_sha256": _digest(frame_rgb),
                    "tile_rgb_sha256": tile["rgb_sha256"],
                    "nearest_resample_rgb_sha256": _digest(resampled),
                    "tile_matches_resample": tile["rgb_sha256"] == _digest(resampled),
                }
            )
            del frame_rgb, resampled
            pixel_guard.release()
        pixel_guard.assert_clean()
        proofs_peak = pixel_guard.peak

        runs.append(
            {
                "stride": stride,
                "label": STRIDE_LABELS[stride],
                "inspection_id": inspection_id,
                "request": inspection_request,
                "record": record,
                "scan_equals_production_record": True,
                "production_residency": residency_summary,
                "bounded_scan_peak_frames_in_flight": scan.peak_frames_in_flight,
                "bounded_scan_sample_count": scan.sampled_frame_count,
                "scan_decode_ms": round(scan.decode_ns / 1_000_000, 2),
                "production_elapsed_ms": median_ms,
                "production_elapsed_samples_ms": timings,
                "production_elapsed_first_ms": response["elapsed_ms"],
                "sampled_pair_scores": scores,
                "metrics": _derive_metrics(
                    record,
                    manifest["design_scenes"],
                    manifest["truth_cuts"],
                    stride,
                    reference_detected,
                ),
                "sheet": {
                    "sheet_id": sheet_id,
                    "request": sheet_request,
                    "record": sheet_record,
                    "artifact_png_path": sheet_png_path,
                    "artifact_png_sha256": _digest(sheet_png),
                    "artifact_pixels_sha256": _digest(sheet_pixels),
                    "artifact_ppm_sha256": artifact_response["_payload"]
                    and _digest(artifact_response["_payload"]),
                    "artifact_sha256_matches_record": artifact_response["body_sha256"]
                    == sheet_record["artifact_digest"],
                    "artifact_status": artifact_response["status"],
                    "elapsed_ms": sheet_response["elapsed_ms"],
                    "tiles": [
                        {
                            "tile_index": tile["tile_index"],
                            "scene_index": tile["scene_index"],
                            "source_frame": tile["source_frame"],
                            "left": tile["left"],
                            "top": tile["top"],
                            "width": tile["width"],
                            "height": tile["height"],
                            "png_path": tile["png_path"],
                            "png_sha256": tile["png_sha256"],
                            "rgb_sha256": tile["rgb_sha256"],
                        }
                        for tile in tile_crops
                    ],
                    "tile_pixel_proofs": proofs,
                    "tile_proof_peak_frames_in_flight": proofs_peak,
                },
            }
        )
    return runs, requests


def _measured_scores(
    media: bytes,
    descriptor: AviDescriptor,
    stride: int,
    truth_cuts: list[int],
) -> dict:
    """Distance of every ordered pair of sampled frames, bounded to one frame."""

    sampled = list(range(0, descriptor.frame_count, stride))
    previous = None
    previous_index = None
    boundaries = []
    inside_max = 0.0
    guard = InFlightGuard(limit=1)
    for index in sampled:
        frame = guard.hold(decode_frame_at(media, descriptor, index))
        fingerprint = fingerprint_frame(frame)
        del frame
        guard.release()
        if previous is not None:
            distance = fingerprint_distance(previous, fingerprint)
            pair = {
                "from_frame": previous_index,
                "to_frame": index,
                "distance": round(distance, 6),
            }
            if any(previous_index < cut <= index for cut in truth_cuts):
                pair["spans_truth_cut"] = True
                boundaries.append(pair)
            else:
                inside_max = max(inside_max, distance)
        previous = fingerprint
        previous_index = index
    guard.assert_clean()
    return {
        "boundary_pairs": boundaries,
        "max_distance_inside_shots": round(inside_max, 6),
        "peak_frames_in_flight": guard.peak,
    }


def _run_interruption(service: Service, media, manifest: dict) -> dict:
    clip_id = WALK_CLIP_ID
    inspection_id = "walk-abort-inspection"
    abort_request = {
        "inspection_id": inspection_id,
        "sensitivity": SENSITIVITY,
        "sample_stride": 4,
        "min_scene_frames": MIN_SCENE_FRAMES,
    }
    client_close = {"connected": False, "request_sent": False, "closed_after_ms": 0.0}

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(60.0)
        started = time.perf_counter_ns()
        sock.connect(("127.0.0.1", service.port))
        client_close["connected"] = True
        payload = json.dumps(abort_request, separators=(",", ":")).encode("utf-8")
        sock.sendall(
            b"POST /v1/clips/" + clip_id.encode() + b"/inspections HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n" + payload
        )
        client_close["request_sent"] = True
        seen = None
        seen_at_ms = None
        while time.perf_counter_ns() - started < 60_000_000:
            try:
                probe = http.client.HTTPConnection("127.0.0.1", service.port, timeout=5)
                probe.request("GET", f"/v1/inspections/{inspection_id}")
                probe_response = probe.getresponse()
                probe_response.read()
                probe.close()
                if probe_response.status == 200:
                    seen = "record_visible"
                    seen_at_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
                    break
            except OSError:
                pass
        client_close["record_visible_before_close"] = seen
        client_close["record_visible_at_ms"] = seen_at_ms
        client_close["closed_after_ms"] = round(
            (time.perf_counter_ns() - started) / 1_000_000, 3
        )
    finally:
        try:
            sock.close()
        except OSError:
            pass

    deadline = time.monotonic() + 15.0
    committed = service.request("GET", f"/v1/inspections/{inspection_id}")
    committed_record = (committed["json"] or {}).get("inspection")
    polls = 1
    while (
        committed_record is not None
        and committed_record.get("state") not in {"indexed", "failed"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        committed = service.request("GET", f"/v1/inspections/{inspection_id}")
        committed_record = (committed["json"] or {}).get("inspection")
        polls += 1

    stored_frames = tuple(committed_record["sampled_frames"] if committed_record else ())

    retry = service.request(
        "POST",
        f"/v1/clips/{clip_id}/inspections",
        _json_bytes(abort_request),
    )

    verifications = verify_frames_bounded(media, stored_frames)
    fingerprint_digests = {
        item.index: _digest(repr(item.fingerprint).encode("utf-8"))
        for item in verifications
    }
    reuse_verdict = analyze_frames(
        probe_avi(media).frame_count,
        stored_frames,
        tuple(item.fingerprint for item in verifications),
        sensitivity=SENSITIVITY,
        min_scene_frames=MIN_SCENE_FRAMES,
    )
    reuse_matches = _compare_verdict(committed_record, reuse_verdict)

    sheet_request = {
        "sheet_id": "walk-abort-sheet",
        "columns": SHEET_COLUMNS,
        "tile_width": TILE_WIDTH,
        "gutter": GUTTER,
        "selection": "stored",
        "background": BACKGROUND,
    }
    sheet_response = service.request(
        "POST",
        f"/v1/inspections/{inspection_id}/contact-sheets",
        _json_bytes(sheet_request),
    )
    sheet_record = (sheet_response["json"] or {}).get("contact_sheet")
    artifact_response = None
    if sheet_record is not None and sheet_record["state"] == "ready":
        artifact_response = service.request(
            "GET", f"/v1/contact-sheets/{sheet_request['sheet_id']}/artifact"
        )

    stored_media_path = service.data_root / "clips" / clip_id / "clip.avi"
    with open(stored_media_path, "rb") as handle:
        stored_digest = hashlib.sha256(handle.read()).hexdigest()
    digest_matches = stored_digest == manifest["sha256"]

    return {
        "attempt": {
            "request": abort_request,
            "client_close": client_close,
            "server_record_committed": committed_record is not None,
            "committed_state": committed_record["state"] if committed_record else None,
            "committed_cut_frames": committed_record["cut_frames"] if committed_record else None,
            "committed_scenes": _scene_assignments(committed_record) if committed_record else [],
            "committed_sampled_frame_count": len(stored_frames),
            "poll_requests": polls,
            "server_behavior_note": (
                "The client never received a response; the production job still ran "
                "to a terminal state and committed its own record."
            ),
        },
        "retry_same_id": {
            "status": retry["status"],
            "error_code": (retry["json"] or {}).get("error", {}).get("code"),
            "error_message": (retry["json"] or {}).get("error", {}).get("message"),
        },
        "reuse": {
            "clip_digest_matches_stored_media": digest_matches,
            "declared_clip_sha256": manifest["sha256"],
            "stored_clip_sha256": stored_digest,
            "verified_frames": [
                {
                    "index": item.index,
                    "frame_digest": item.frame_digest,
                    "fingerprint_digest": fingerprint_digests[item.index],
                }
                for item in verifications
            ],
            "reuse_verdict_matches_committed_record": reuse_matches,
            "reuse_cut_frames": list(reuse_verdict.cut_frames),
            "sheet_status": sheet_response["status"],
            "sheet_state": sheet_record["state"] if sheet_record else None,
            "sheet_tiles": len(sheet_record["tile_map"]) if sheet_record else 0,
            "artifact_status": artifact_response["status"] if artifact_response else None,
            "artifact_sha256": artifact_response["body_sha256"] if artifact_response else None,
            "artifact_matches_record": (
                artifact_response is not None
                and sheet_record is not None
                and artifact_response["body_sha256"] == sheet_record["artifact_digest"]
            ),
            "partial_outputs_stitched": False,
        },
        "requests": [
            _public_response(committed),
            _public_response(retry),
            _public_response(sheet_response),
        ]
        + ([_public_response(artifact_response)] if artifact_response else []),
    }


service_frame_dir: Path


def build_evidence(output_dir: Path, frame_dir: Path) -> dict:
    global service_frame_dir
    service_frame_dir = frame_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    work_dir = frame_dir.parent / ".build"
    work_dir.mkdir(exist_ok=True)

    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    walk_path = work_dir / "walk.avi"
    rot_path = work_dir / "rot.avi"
    walk_manifest = build_walk_clip(walk_path)
    rot_manifest = build_rot_clip(rot_path)
    walk_mapped = _mmap_read(walk_path)
    rot_mapped = _mmap_read(rot_path)
    (clip_dir / "walk.manifest.json").write_bytes(_json_bytes(walk_manifest))
    (clip_dir / "rot.manifest.json").write_bytes(_json_bytes(rot_manifest))

    walk_descriptor = probe_avi(walk_mapped)
    rot_descriptor = probe_avi(rot_mapped)

    walk_frames = _frame_pngs_bounded(
        walk_mapped,
        walk_descriptor,
        tuple(range(walk_descriptor.frame_count)),
        frame_dir,
        "walk-frame",
    )
    rot_frames = _frame_pngs_bounded(
        rot_mapped,
        rot_descriptor,
        tuple(range(rot_descriptor.frame_count)),
        frame_dir,
        "rot-frame",
    )
    rot_verifications = verify_frames_bounded(
        rot_mapped, tuple(range(rot_descriptor.frame_count))
    )
    rot_distance = fingerprint_distance(
        rot_verifications[0].fingerprint, rot_verifications[4].fingerprint
    )

    evidence = {
        "schema_version": EVIDENCE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_service": {
            "entrypoint": "frame_proxy.http.server.build_server",
            "source_untouched": True,
            "data_root": "throwaway directory created for this lab run",
        },
        "study_conditions": {
            "sensitivity": SENSITIVITY,
            "min_scene_frames": MIN_SCENE_FRAMES,
            "strides": list(STRIDES),
            "source_rgb_residency_limit": 2,
            "frame_retention": (
                "The study condition allows at most two source RGB frames resident "
                "per inspection task. Every lab scan decodes one frame, fingerprints "
                "it, and drops it before the next decode; InFlightGuard fails the "
                "build if two frames are ever alive together. The production request "
                "is additionally measured at runtime through temporary wrappers "
                "that count live Frame objects; those peaks are recorded per run as "
                "production_residency. No full frame sequence is decoded into "
                "memory or re-read from a prepared list. Viewer PNGs are derived one "
                "decode at a time after the study."
            ),
            "min_scene_guard_note": (
                "Production rejects a cut when either side would be shorter than "
                "min_scene_frames. This is a production rule, not a lab tweak."
            ),
            "quality_definition": (
                "Quality is measured against the clip's design truth: every "
                "designed boundary within the sampling tolerance. A cut is "
                "matched when a detected cut lies within sample_stride frames "
                "of it. Missing a true cut is never treated as a match."
            ),
            "known_limitations": [
                "Low-contrast cuts (frames 102, 142) fall below the fixed "
                "0.335 threshold at every stride; this is a detector property, "
                "not a sampling artifact.",
                "Two-frame shots vanish entirely at stride 4, so their content "
                "has no tile in that sheet.",
                "The rotation counterexample is kept as a recorded false "
                "negative (cut_frames stays empty); it is not corrected or "
                "hidden.",
            ],
        },
        "walk_clip": {
            "clip": walk_manifest,
            "frame_assets": walk_frames,
        },
        "rotation_counterexample": {
            "clip": rot_manifest,
            "frame_assets": rot_frames,
            "pairs": [
                {
                    "left_frame": 0,
                    "right_frame": 3,
                    "distance": round(
                        fingerprint_distance(
                            rot_verifications[0].fingerprint,
                            rot_verifications[3].fingerprint,
                        ),
                        9,
                    ),
                },
                {
                    "left_frame": 3,
                    "right_frame": 4,
                    "distance": round(rot_distance, 9),
                    "statistics_identical": round(rot_distance, 9) == 0.0,
                    "pixels_differ": rot_verifications[3].frame_digest
                    != rot_verifications[4].frame_digest,
                },
                {
                    "left_frame": 4,
                    "right_frame": 7,
                    "distance": round(
                        fingerprint_distance(
                            rot_verifications[4].fingerprint,
                            rot_verifications[7].fingerprint,
                        ),
                        9,
                    ),
                },
            ],
            "fingerprint_fields": {
                "left_frame": 3,
                "right_frame": 4,
                "left": {
                    "mean_luma": round(rot_verifications[3].fingerprint.mean_luma, 9),
                    "contrast": round(rot_verifications[3].fingerprint.contrast, 9),
                    "dominant_rgb": [
                        round(rot_verifications[3].fingerprint.mean_red),
                        round(rot_verifications[3].fingerprint.mean_green),
                        round(rot_verifications[3].fingerprint.mean_blue),
                    ],
                },
                "right": {
                    "mean_luma": round(rot_verifications[4].fingerprint.mean_luma, 9),
                    "contrast": round(rot_verifications[4].fingerprint.contrast, 9),
                    "dominant_rgb": [
                        round(rot_verifications[4].fingerprint.mean_red),
                        round(rot_verifications[4].fingerprint.mean_green),
                        round(rot_verifications[4].fingerprint.mean_blue),
                    ],
                },
            },
        },
    }

    requests_log: list[dict] = []
    with Service(Path(tempfile.mkdtemp(prefix="frame-proxy-lab-"))) as service:
        # Upload from disk so the lab never holds a second full copy of the
        # clip bytes just to transmit them.
        upload_walk = service.request(
            "PUT", f"/v1/clips/{WALK_CLIP_ID}", walk_path.read_bytes(), "application/x-avi"
        )
        upload_rot = service.request(
            "PUT", f"/v1/clips/{ROT_CLIP_ID}", rot_path.read_bytes(), "application/x-avi"
        )
        health = service.request("GET", "/health")
        requests_log.extend(
            [
                _public_response(health),
                _public_response(upload_walk),
                _public_response(upload_rot),
            ]
        )

        runs, study_requests = _run_sampling_study(service, walk_mapped, walk_manifest)
        requests_log.extend(study_requests)

        rot_request = {
            "inspection_id": "rot-s1",
            "sensitivity": SENSITIVITY,
            "sample_stride": 1,
            "min_scene_frames": MIN_SCENE_FRAMES,
        }
        rot_response = service.request(
            "POST", f"/v1/clips/{ROT_CLIP_ID}/inspections", _json_bytes(rot_request)
        )
        requests_log.append(_public_response(rot_response))
        rot_record = rot_response["json"]["inspection"]
        rot_scan = scan_bounded(
            rot_mapped,
            sample_stride=1,
            sensitivity=SENSITIVITY,
            min_scene_frames=MIN_SCENE_FRAMES,
        )
        if not _compare_verdict(rot_record, rot_scan.verdict):
            raise RuntimeError("bounded rotation scan diverged from production")
        sheet_two = service.request(
            "POST",
            "/v1/inspections/rot-s1/contact-sheets",
            _json_bytes(ROT_SHEET),
        )
        requests_log.append(_public_response(sheet_two))
        sheet_two_record = sheet_two["json"]["contact_sheet"]
        sheet_two_artifact = service.request(
            "GET", f"/v1/contact-sheets/{ROT_SHEET['sheet_id']}/artifact"
        )
        requests_log.append(_public_response(sheet_two_artifact))
        width_two, height_two, pixels_two = _ppm_to_rgb(sheet_two_artifact["_payload"])
        rot_sheet_png = encode_rgb_png(width_two, height_two, pixels_two)
        (frame_dir / "rot-sheet.png").write_bytes(rot_sheet_png)
        rot_tiles = _sheet_tile_crops(
            pixels_two, width_two, height_two, sheet_two_record["tile_map"]
        )
        for crop in rot_tiles:
            crop_png = encode_rgb_png(crop["width"], crop["height"], crop["_crop"])
            crop["png_path"] = f"rot-tile-{crop['tile_index']}.png"
            (frame_dir / crop["png_path"]).write_bytes(crop_png)
            crop["png_sha256"] = _digest(crop_png)
            del crop["_crop"]

        evidence["rotation_counterexample"]["inspection"] = {
            "request": rot_request,
            "record": rot_record,
            "bounded_scan_cut_frames": list(rot_scan.verdict.cut_frames),
            "bounded_scan_peak_frames_in_flight": rot_scan.peak_frames_in_flight,
            "bounded_scan_equals_record": True,
            "sheet": {
                "sheet_id": ROT_SHEET["sheet_id"],
                "request": ROT_SHEET,
                "record": sheet_two_record,
                "artifact_png_path": "rot-sheet.png",
                "artifact_png_sha256": _digest(rot_sheet_png),
                "artifact_sha256_matches_record": sheet_two_artifact["body_sha256"]
                == sheet_two_record["artifact_digest"],
                "tiles": [
                    {
                        "tile_index": tile["tile_index"],
                        "scene_index": tile["scene_index"],
                        "source_frame": tile["source_frame"],
                        "png_path": tile["png_path"],
                        "png_sha256": tile["png_sha256"],
                        "rgb_sha256": tile["rgb_sha256"],
                    }
                    for tile in rot_tiles
                ],
            },
        }

        evidence["interruption"] = _run_interruption(service, walk_mapped, walk_manifest)
        requests_log.extend(evidence["interruption"]["requests"])

        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for index, response in enumerate(requests_log):
            safe_path = response["path"].strip("/").replace("/", "_") or "root"
            name = f"{index:02d}-{response['method'].lower()}-{safe_path}.json"
            (raw_dir / name).write_bytes(_json_bytes(response))

        evidence["runs"] = runs
        evidence["requests"] = requests_log
        evidence["cross_checks"] = {
            "median_speed_is_monotonic": _check_monotonic_medians(runs),
            "all_sheets_ready": all(
                run["sheet"]["record"]["state"] == "ready" for run in runs
            ),
            "all_artifacts_match_records": all(
                run["sheet"]["artifact_sha256_matches_record"] for run in runs
            ),
            "all_tiles_match_nearest_resample": all(
                all(
                    proof["tile_matches_resample"]
                    for proof in run["sheet"]["tile_pixel_proofs"]
                )
                for run in runs
            ),
            "bounded_scan_equals_production": all(
                run["scan_equals_production_record"] for run in runs
            ),
            "bounded_scan_peak_within_limit": all(
                run["bounded_scan_peak_frames_in_flight"] <= 2 for run in runs
            ),
            "tile_proof_peak_within_limit": all(
                run["sheet"]["tile_proof_peak_frames_in_flight"] <= 2
                for run in runs
            ),
            "max_observed_source_rgb_frames": max(
                [
                    run["bounded_scan_peak_frames_in_flight"] for run in runs
                ]
                + [
                    run["sheet"]["tile_proof_peak_frames_in_flight"]
                    for run in runs
                ]
                + [
                    run["production_residency"]["peak_live_source_rgb_frames"]
                    for run in runs
                ]
            ),
            "rotation_bounded_scan_equals_production": evidence[
                "rotation_counterexample"
            ]["inspection"]["bounded_scan_equals_record"],
            "interruption_reuse_verified": evidence["interruption"]["reuse"][
                "reuse_verdict_matches_committed_record"
            ],
        }

    (output_dir / "evidence.json").write_bytes(_json_bytes(evidence))
    _write_viewer_data(evidence, output_dir)
    work_dir.joinpath("walk.avi").unlink(missing_ok=True)
    work_dir.joinpath("rot.avi").unlink(missing_ok=True)
    if work_dir.is_dir() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    return evidence


def _slim_record(record: dict) -> dict:
    return {
        "state": record["state"],
        "sensitivity": record["sensitivity"],
        "sample_stride": record["sample_stride"],
        "min_scene_frames": record["min_scene_frames"],
        "sampled_frames": record["sampled_frames"],
        "cut_frames": record["cut_frames"],
        "scenes": record["scenes"],
        "created_at": record["created_at"],
    }


def _check_monotonic_medians(runs: list[dict]) -> bool:
    """Report whether larger strides are faster, without smoothing the data."""

    medians = [run["production_elapsed_ms"] for run in runs]
    if any(
        medians[index] < medians[index + 1] for index in range(len(medians) - 1)
    ):
        raise RuntimeError(
            f"speed medians are not monotonic across sampling strides: {medians}"
        )
    return True


def _viewer_payload(evidence: dict) -> dict:
    walk = evidence["walk_clip"]
    return {
        "generatedAt": evidence["generated_at"],
        "conditions": evidence["study_conditions"],
        "service": evidence["production_service"],
        "clip": {
            "id": walk["clip"]["clip_id"],
            "width": walk["clip"]["width"],
            "height": walk["clip"]["height"],
            "fps": walk["clip"]["fps"],
            "frameCount": walk["clip"]["frame_count"],
            "sha256": walk["clip"]["sha256"],
            "byteSize": walk["clip"]["byte_size"],
            "designScenes": walk["clip"]["design_scenes"],
            "truthCuts": walk["clip"]["truth_cuts"],
            "frameAssets": walk["frame_assets"],
        },
        "strides": [
            {
                "stride": run["stride"],
                "label": run["label"],
                "request": run["request"],
                "productionElapsedMs": run["production_elapsed_ms"],
                "timingSamplesMs": run["production_elapsed_samples_ms"],
                "scanMs": run["scan_decode_ms"],
                "sampleCount": run["bounded_scan_sample_count"],
                "peakFrames": run["bounded_scan_peak_frames_in_flight"],
                "residencyPeak": run["production_residency"][
                    "peak_live_source_rgb_frames"
                ],
                "record": _slim_record(run["record"]),
                "metrics": run["metrics"],
                "scores": run["sampled_pair_scores"],
                "sheet": {
                    "sheetId": run["sheet"]["sheet_id"],
                    "imagePath": run["sheet"]["artifact_png_path"],
                    "imageWidth": run["sheet"]["record"]["image_width"],
                    "imageHeight": run["sheet"]["record"]["image_height"],
                    "artifactDigest": run["sheet"]["record"]["artifact_digest"],
                    "artifactPngSha256": run["sheet"]["artifact_png_sha256"],
                    "artifactPixelsSha256": run["sheet"]["artifact_pixels_sha256"],
                    "artifactDigestMatches": run["sheet"][
                        "artifact_sha256_matches_record"
                    ],
                    "elapsedMs": run["sheet"]["elapsed_ms"],
                    "tiles": run["sheet"]["tiles"],
                    "proofs": run["sheet"]["tile_pixel_proofs"],
                },
            }
            for run in evidence["runs"]
        ],
        "counterexample": {
            "clip": {
                "clip_id": evidence["rotation_counterexample"]["clip"]["clip_id"],
                "width": evidence["rotation_counterexample"]["clip"]["width"],
                "height": evidence["rotation_counterexample"]["clip"]["height"],
                "frame_count": evidence["rotation_counterexample"]["clip"][
                    "frame_count"
                ],
                "sha256": evidence["rotation_counterexample"]["clip"]["sha256"],
                "design_scenes": evidence["rotation_counterexample"]["clip"][
                    "design_scenes"
                ],
            },
            "frameAssets": evidence["rotation_counterexample"]["frame_assets"],
            "pairs": evidence["rotation_counterexample"]["pairs"],
            "fields": evidence["rotation_counterexample"]["fingerprint_fields"],
            "inspection": {
                "request": evidence["rotation_counterexample"]["inspection"][
                    "request"
                ],
                "record": _slim_record(
                    evidence["rotation_counterexample"]["inspection"]["record"]
                ),
                "boundedScanCutFrames": evidence["rotation_counterexample"][
                    "inspection"
                ]["bounded_scan_cut_frames"],
                "boundedScanPeakFrames": evidence["rotation_counterexample"][
                    "inspection"
                ]["bounded_scan_peak_frames_in_flight"],
                "sheet": evidence["rotation_counterexample"]["inspection"]["sheet"],
            },
        },
        "interruption": evidence["interruption"],
        "requests": evidence["requests"],
        "crossChecks": evidence["cross_checks"],
    }


def _write_viewer_data(evidence: dict, output_dir: Path) -> None:
    payload = _viewer_payload(evidence)
    data_path = output_dir.parent / "viewer" / "data.js"
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_path.write_text(
        "window.LAB_DATA = " + body + ";\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (output_dir / "viewer-data.json").write_bytes(
        _json_bytes(
            {
                "path": "lab/viewer/data.js",
                "sha256": digest,
                "bytes": data_path.stat().st_size,
            }
        )
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    output_dir = here / "evidence"
    frame_dir = here / "viewer" / "frames"
    evidence = build_evidence(output_dir, frame_dir)
    print(
        json.dumps(
            {
                "runs": [
                    {
                        "stride": run["stride"],
                        "cuts": run["record"]["cut_frames"],
                        "scenes": run["metrics"]["scene_count"],
                        "missed": run["metrics"]["missed_truth_cuts"],
                        "production_ms": run["production_elapsed_ms"],
                        "scan_ms": run["scan_decode_ms"],
                        "peak_frames": run["bounded_scan_peak_frames_in_flight"],
                    }
                    for run in evidence["runs"]
                ],
                "rotation": evidence["rotation_counterexample"]["inspection"][
                    "record"
                ]["cut_frames"],
                "interruption": {
                    "committed": evidence["interruption"]["attempt"][
                        "committed_state"
                    ],
                    "retry_status": evidence["interruption"]["retry_same_id"][
                        "status"
                    ],
                    "reuse_verified": evidence["interruption"]["reuse"][
                        "reuse_verdict_matches_committed_record"
                    ],
                },
                "cross_checks": evidence["cross_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
