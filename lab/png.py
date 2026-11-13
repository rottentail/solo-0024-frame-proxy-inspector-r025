"""Minimal, dependency-free PNG writer used only for the lab viewer.

The lab converts production PPM artifacts and decoded RGB frames into PNG so a
browser can display them. The conversion is lossless: the source pixel buffer
is written byte-for-byte with no resampling or color management.
"""

from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_rgb_png(width: int, height: int, pixels: bytes) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    if len(pixels) != width * height * 3:
        raise ValueError("PNG pixel length does not match its dimensions")

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = bytearray()
    stride = width * 3
    for row in range(height):
        rows.append(0)  # filter type 0: no filter, pixels stay verbatim
        start = row * stride
        rows.extend(pixels[start : start + stride])
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _chunk(b"IHDR", header)
    payload += _chunk(b"IDAT", zlib.compress(bytes(rows), 9))
    payload += _chunk(b"IEND", b"")
    return payload


def decode_rgb_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode the subset of PNG written by encode_rgb_png.

    Used by the package self-check to prove the PNG display copies preserve the
    exact pixel buffer they were generated from.
    """

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG payload")
    cursor = 8
    width = 0
    height = 0
    idat = bytearray()
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("PNG chunk header is truncated")
        length = struct.unpack_from(">I", data, cursor)[0]
        tag = data[cursor + 4 : cursor + 8]
        payload = data[cursor + 8 : cursor + 8 + length]
        if len(payload) != length:
            raise ValueError("PNG chunk payload is truncated")
        cursor += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("unsupported PNG header")
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError("PNG has no usable header")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("PNG scanline length does not match its header")
    pixels = bytearray()
    for row in range(height):
        start = row * (stride + 1)
        if raw[start] != 0:
            raise ValueError("only unfiltered PNG rows are supported")
        pixels.extend(raw[start + 1 : start + 1 + stride])
    return width, height, bytes(pixels)
