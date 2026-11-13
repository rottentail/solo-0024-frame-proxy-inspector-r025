"""Measure source RGB frame residency of a production call at runtime.

The production service is imported, not edited. To report how many decoded
source RGB frames a production request keeps alive, the lab temporarily wraps
the ``decode_frame_at`` name *inside the production module's namespace* and
counts live ``Frame`` objects with the garbage collector. Wrappers are removed
in ``finally`` blocks. The wrapper changes no production behavior: it calls the
original decoder and returns its original result.

The lab's own scan loops do not need this probe; they use the explicit
``InFlightGuard`` in ``lab.sampling`` which refuses to exceed its limit.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager

from frame_proxy.media.frames import Frame


class ResidencyProbe:
    def __init__(self) -> None:
        self.calls = 0
        self.peak = 0
        self._original = None
        self._module = None
        self._attribute = ""

    def _live_frames(self) -> int:
        return sum(1 for item in gc.get_objects() if isinstance(item, Frame))

    def wrapped(self, buffer, descriptor, frame_index):
        frame = self._original(buffer, descriptor, frame_index)
        self.calls += 1
        live = self._live_frames()
        self.peak = max(self.peak, live)
        return frame


@contextmanager
def probe_residency(module, attribute: str = "decode_frame_at"):
    """Temporarily wrap a decoder name in a module and record live frames."""

    probe = ResidencyProbe()
    probe._module = module
    probe._attribute = attribute
    probe._original = getattr(module, attribute)
    setattr(module, attribute, probe.wrapped)
    try:
        yield probe
    finally:
        setattr(module, attribute, probe._original)


def summarize(probe: ResidencyProbe, source_frame_count: int) -> dict:
    return {
        "decode_calls": probe.calls,
        "peak_live_source_rgb_frames": probe.peak,
        "source_rgb_frame_bytes": 160 * 90 * 3,
        "source_frame_count_sampled": source_frame_count,
        "interpretation": (
            "Peak count of live Frame objects observed while the production "
            "request was running. 1 means the request decoded and released "
            "frames one at a time."
        ),
    }
