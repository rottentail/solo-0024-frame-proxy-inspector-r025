"""Deterministic source footage and its design manifest for the lab.

The production service only accepts real RIFF/AVI footage. This module writes
the two lab clips with plain ``struct`` and appends every frame chunk to the
open file as soon as its payload is built. One generated RGB frame and at
most one rotation copy exist at a time; no list of frame payloads is
assembled, either in memory or on disk before publication.

Clip content is designed around observable situations:

* ``lab-walk-01`` walks through nine designed shots. Two shots are only two
  frames long; one hard cut is deliberately low-contrast. A marker moves
  inside every shot so frame stepping is visible.
* ``lab-rot-01`` repeats one frame four times, then repeats its exact
  180-degree rotation four times. Fingerprints cannot tell the halves apart,
  while a human immediately sees the flipped marker.
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

WIDTH = 160
HEIGHT = 90
FPS = 12
ROT_FRAME_COUNT = 8

WALK_CLIP_ID = "lab-walk-01"
ROT_CLIP_ID = "lab-rot-01"


def _scene_spec(label: str, bg, accent, length: int, note: str = "") -> dict:
    return {
        "label": label,
        "background_rgb": list(bg),
        "accent_rgb": list(accent),
        "note": note,
        "length": length,
    }


SCENE_SPECS = (
    _scene_spec("冷青阔景", (18, 44, 58), (232, 176, 52), 42, "42 帧，主场景"),
    _scene_spec("白闪短镜头", (250, 244, 230), (250, 60, 40), 2, "仅两帧；步长 4 完全跳过"),
    _scene_spec("深蓝棚景", (36, 52, 120), (120, 220, 255), 30, "深蓝主色"),
    _scene_spec("暗绿硬切", (44, 96, 52), (230, 240, 190), 28, "统计指纹与前一镜头距离很小"),
    _scene_spec("洋红场景 A", (102, 42, 86), (245, 190, 120), 8, "洋红前段"),
    _scene_spec("暗底橙闪", (12, 10, 8), (250, 230, 60), 2, "仅两帧；步长 4 完全跳过"),
    _scene_spec("洋红场景 B", (102, 42, 86), (245, 190, 120), 30, "与场景 A 同色，视觉连续"),
    _scene_spec("中性灰", (70, 70, 74), (250, 250, 250), 24, "中性灰底"),
    _scene_spec("橄榄场景", (60, 70, 28), (200, 230, 90), 34, "收尾场景"),
)

MARKER_WIDTH = 16
MARKER_HEIGHT = 9


def _scene_base(bg, accent) -> bytes:
    base = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if x < 34 and y < 32:
                color = accent
            elif (x + y) % 37 < 4:
                color = tuple(min(255, value + 28) for value in bg)
            else:
                color = bg
            offset = (y * WIDTH + x) * 3
            base[offset : offset + 3] = bytes(color)
    return bytes(base)


def _with_marker(base: bytes, marker_x: int) -> bytes:
    top = HEIGHT // 2 - MARKER_HEIGHT // 2
    frame = bytearray(base)
    color = (240, 240, 235)
    for y in range(top, top + MARKER_HEIGHT):
        for x in range(marker_x, marker_x + MARKER_WIDTH):
            offset = (y * WIDTH + x) * 3
            frame[offset : offset + 3] = bytes(color)
    return bytes(frame)


def _rot180(frame: bytes) -> bytes:
    rotated = bytearray(WIDTH * HEIGHT * 3)
    stride = WIDTH * 3
    for y in range(HEIGHT):
        target_row = (HEIGHT - 1 - y) * stride
        for x in range(WIDTH):
            source = target_row + (WIDTH - 1 - x) * 3
            destination = y * stride + x * 3
            rotated[destination : destination + 3] = frame[source : source + 3]
    return bytes(rotated)


def _rgb_to_bgr_bottom_up(frame: bytes) -> bytes:
    stride = WIDTH * 3
    rows = []
    for y in range(HEIGHT - 1, -1, -1):
        row = bytearray(frame[y * stride : (y + 1) * stride])
        row[0::3], row[2::3] = row[2::3], row[0::3]
        rows.append(bytes(row))
    return b"".join(rows)


def _header_bytes(frame_count: int) -> bytes:
    row_stride = WIDTH * 3
    frame_size = row_stride * HEIGHT
    avih = struct.pack(
        "<14I",
        1_000_000 // FPS,
        frame_size * FPS,
        0,
        0x10,
        frame_count,
        0,
        1,
        frame_size,
        WIDTH,
        HEIGHT,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        FPS,
        0,
        frame_count,
        frame_size,
        0,
        frame_size,
        0,
        0,
        WIDTH,
        HEIGHT,
    )
    strf = struct.pack(
        "<IiiHHIIiiII",
        40,
        WIDTH,
        HEIGHT,
        1,
        24,
        0,
        frame_size,
        2835,
        2835,
        0,
        0,
    )
    stream_body = b"strl" + _chunk(b"strh", strh) + _chunk(b"strf", strf)
    stream = b"LIST" + struct.pack("<I", len(stream_body)) + stream_body
    header_body = b"hdrl" + _chunk(b"avih", avih) + stream
    return b"LIST" + struct.pack("<I", len(header_body)) + header_body


def _chunk(tag: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) & 1 else b""
    return tag + struct.pack("<I", len(payload)) + payload + padding


def _write_container(
    path: Path,
    frame_count: int,
    payload_source,
) -> None:
    """Stream a container to disk, one generated frame at a time.

    The RIFF and movi sizes are computed from the fixed frame geometry, so the
    writer can append each frame chunk immediately instead of buffering the
    whole clip. Only the frame currently being generated is resident.
    """

    header_body = _header_bytes(frame_count)
    frame_size = WIDTH * 3 * HEIGHT
    movi_size = 4 + frame_count * (8 + frame_size)
    riff_size = 4 + len(header_body) + 8 + movi_size
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", riff_size) + b"AVI ")
        handle.write(header_body)
        handle.write(b"LIST" + struct.pack("<I", movi_size) + b"movi")
        for _ in range(frame_count):
            payload = payload_source()
            if len(payload) != frame_size:
                raise ValueError("generated frame payload has the wrong size")
            handle.write(b"00db" + struct.pack("<I", frame_size))
            handle.write(payload)
            del payload
        handle.flush()
        os.fsync(handle.fileno())
    if path.stat().st_size != riff_size + 8:
        raise ValueError("streamed AVI size accounting is inconsistent")


def build_walk_clip(path: Path) -> dict:
    scenes = []
    cursor = 0
    for index, spec in enumerate(SCENE_SPECS):
        length = int(spec["length"])
        scenes.append(
            {
                "index": index,
                "start_frame": cursor,
                "end_frame": cursor + length,
                "length": length,
                "label": spec["label"],
                "note": spec["note"],
                "background_rgb": spec["background_rgb"],
                "accent_rgb": spec["accent_rgb"],
            }
        )
        cursor += length
    if cursor != 200:
        raise ValueError("walk clip design must cover exactly 200 frames")

    state = {"scene": 0, "frame_in_scene": 0, "base": b"", "base_scene": -1}

    def payload_source() -> bytes:
        while state["scene"] < len(SCENE_SPECS) and state["frame_in_scene"] >= int(
            SCENE_SPECS[state["scene"]]["length"]
        ):
            state["scene"] += 1
            state["frame_in_scene"] = 0
        if state["scene"] >= len(SCENE_SPECS):
            raise ValueError("walk clip generator ran past its design")
        spec = SCENE_SPECS[state["scene"]]
        if state["base_scene"] != state["scene"]:
            state["base"] = _scene_base(
                tuple(spec["background_rgb"]), tuple(spec["accent_rgb"])
            )
            state["base_scene"] = state["scene"]
        length = int(spec["length"])
        step = (WIDTH - 32 - 8) / max(1, length - 1)
        marker_x = 8 + int(state["frame_in_scene"] * step)
        rgb = _with_marker(state["base"], marker_x)
        state["frame_in_scene"] += 1
        return _rgb_to_bgr_bottom_up(rgb)

    _write_container(path, 200, payload_source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "clip_id": WALK_CLIP_ID,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": 200,
        "byte_size": path.stat().st_size,
        "sha256": digest,
        "design_scenes": scenes,
        "truth_cuts": [scene["start_frame"] for scene in scenes][1:],
        "notes": [
            "Design truth is fixed at generation time and never edited after inspection.",
            "Frame stepping shows the marker walking across each shot.",
        ],
    }


def build_rot_clip(path: Path) -> dict:
    base = _scene_base((22, 26, 34), (210, 54, 44))
    a = _with_marker(base, 18)
    b = _rot180(a)
    state = {"next": 0}

    def payload_source() -> bytes:
        index = state["next"]
        if index >= ROT_FRAME_COUNT:
            raise ValueError("rotation clip generator ran past its design")
        state["next"] += 1
        return _rgb_to_bgr_bottom_up(a if index < 4 else b)

    _write_container(path, ROT_FRAME_COUNT, payload_source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "clip_id": ROT_CLIP_ID,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": ROT_FRAME_COUNT,
        "byte_size": path.stat().st_size,
        "sha256": digest,
        "design_scenes": [
            {
                "index": 0,
                "start_frame": 0,
                "end_frame": 4,
                "length": 4,
                "label": "原方向",
                "note": "标记在左侧",
            },
            {
                "index": 1,
                "start_frame": 4,
                "end_frame": 8,
                "length": 4,
                "label": "180°翻转",
                "note": "整帧逐像素旋转 180°，标记跳到右侧",
            },
        ],
        "truth_cuts": [4],
        "notes": [
            "Frame 4 is the exact 180-degree rotation of frame 0.",
            "Every pixel changes, but every fingerprint statistic is unchanged.",
        ],
    }
