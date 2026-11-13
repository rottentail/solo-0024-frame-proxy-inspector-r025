from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from frame_proxy.media.avi import AviDescriptor, probe_avi
from frame_proxy.media.detector import AnalysisResult, analyze_frames
from frame_proxy.media.fingerprint import FrameFingerprint, fingerprint_frame
from frame_proxy.media.frames import Frame, decode_frame_at


@dataclass(slots=True)
class InFlightGuard:
    """Count live source RGB frames and refuse to exceed the lab's limit.

    ``hold`` is called for every decoded frame, ``release`` when the caller
    drops it. The floor must stay at zero and the peak must stay at one, which
    is stricter than the "no more than two" study condition. A violation
    raises immediately instead of silently weakening the claim.
    """

    limit: int = 1
    live: int = 0
    peak: int = 0
    violations: list[str] = field(default_factory=list)

    def hold(self, frame: Frame) -> Frame:
        self.live += 1
        self.peak = max(self.peak, self.live)
        if self.live > self.limit:
            self.violations.append(
                f"live source RGB frames reached {self.live}, limit is {self.limit}"
            )
            raise RuntimeError(self.violations[-1])
        return frame

    def release(self) -> None:
        self.live -= 1
        if self.live < 0:
            raise RuntimeError("frame guard released more frames than it held")

    def assert_clean(self) -> None:
        if self.live != 0:
            raise RuntimeError("frame guard still holds decoded frames")
        if self.peak > self.limit:
            raise RuntimeError("frame guard observed a retention violation")


@dataclass(frozen=True, slots=True)
class BoundedScan:
    verdict: AnalysisResult
    decode_ns: int
    sampled_frame_count: int
    peak_frames_in_flight: int


def scan_bounded(
    media,
    *,
    sample_stride: int,
    sensitivity: float,
    min_scene_frames: int,
) -> BoundedScan:
    """Scan sampled frames while holding at most one RGB frame at a time.

    Only fingerprints (statistics, not pixels) are retained. The decoded frame
    reference is overwritten on each iteration and the payload is never copied
    into a collection. The guard raises if two frames are ever live together.
    """

    descriptor = probe_avi(media)
    sampled_frames = tuple(range(0, descriptor.frame_count, sample_stride))
    fingerprints: list[FrameFingerprint] = []
    guard = InFlightGuard(limit=1)
    started = time.perf_counter_ns()
    for frame_index in sampled_frames:
        frame = guard.hold(decode_frame_at(media, descriptor, frame_index))
        fingerprints.append(fingerprint_frame(frame))
        del frame
        guard.release()
    elapsed = time.perf_counter_ns() - started
    guard.assert_clean()
    verdict = analyze_frames(
        descriptor.frame_count,
        sampled_frames,
        tuple(fingerprints),
        sensitivity=sensitivity,
        min_scene_frames=min_scene_frames,
    )
    return BoundedScan(
        verdict=verdict,
        decode_ns=elapsed,
        sampled_frame_count=len(sampled_frames),
        peak_frames_in_flight=guard.peak,
    )


@dataclass(frozen=True, slots=True)
class FrameVerification:
    index: int
    frame_digest: str
    width: int
    height: int
    fingerprint: FrameFingerprint


def verify_frames_bounded(
    media,
    frame_indices: tuple[int, ...],
) -> list[FrameVerification]:
    """Digest selected frames one at a time without retaining their pixels."""

    descriptor = probe_avi(media)
    guard = InFlightGuard(limit=1)
    results: list[FrameVerification] = []
    for frame_index in frame_indices:
        frame = guard.hold(decode_frame_at(media, descriptor, frame_index))
        digest = hashlib.sha256(frame.rgb).hexdigest()
        results.append(
            FrameVerification(
                index=frame_index,
                frame_digest=digest,
                width=frame.width,
                height=frame.height,
                fingerprint=fingerprint_frame(frame),
            )
        )
        del frame
        guard.release()
    guard.assert_clean()
    return results


def scan_summary(media, descriptor: AviDescriptor, scan: BoundedScan) -> dict:
    """Confirm the bounded scan reproduces the production inspection verdict."""

    return {
        "cut_frames": list(scan.verdict.cut_frames),
        "scene_ranges": [
            [scene.start_frame, scene.end_frame]
            for scene in scan.verdict.scenes
        ],
        "sampled_frames": list(scan.verdict.sampled_frames),
        "scan_decode_ms": round(scan.decode_ns / 1_000_000, 2),
        "scan_sample_count": scan.sampled_frame_count,
        "scan_peak_frames_in_flight": scan.peak_frames_in_flight,
        "width": descriptor.width,
        "height": descriptor.height,
        "frame_count": descriptor.frame_count,
    }
