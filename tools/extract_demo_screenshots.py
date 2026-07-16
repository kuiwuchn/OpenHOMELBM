"""Extract reproducible documentation screenshots from checked-in demo videos."""

from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = {
    "eel2d.jpg": ("outputs/eel_lbm.mp4", 0.70, 0),
    "eel3d-vorticity.jpg": ("outputs/eel_lbm_orbit_slice9.mp4", 0.70, 0),
    "karman2d.jpg": ("outputs/karman_vortex_2d.mp4", 0.90, 0),
    "sac-forward.jpg": (
        "outputs/sac_minimal/videos/eel2d_forward_policy.mp4",
        0.16,
        0,
    ),
}


def extract_frame(
    video_path: Path,
    fraction: float,
    crop_top: int,
    output_path: Path,
) -> None:
    """Write one JPEG frame selected by its relative position in a video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = round(max(0, frame_count - 1) * fraction)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    if crop_top:
        frame = frame[crop_top:, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Could not write screenshot: {output_path}")
    print(f"{video_path.relative_to(PROJECT_ROOT)} -> {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Regenerate all screenshots used by the README and MkDocs site."""
    output_dir = PROJECT_ROOT / "docs" / "assets" / "demos"
    for output_name, (video_name, fraction, crop_top) in SCREENSHOTS.items():
        extract_frame(
            PROJECT_ROOT / video_name,
            fraction,
            crop_top,
            output_dir / output_name,
        )


if __name__ == "__main__":
    main()
