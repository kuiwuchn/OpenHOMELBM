"""Extract reproducible documentation media from checked-in demo videos."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


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
ANIMATIONS = {
    "eel2d.gif": (
        "outputs/eel_lbm.mp4",
        0.0,
        4.0,
        10.0,
        360,
        128,
        0,
    ),
    "eel3d-vorticity.gif": (
        "outputs/eel_lbm_orbit_slice9.mp4",
        2.0,
        6.0,
        10.0,
        480,
        128,
        0,
    ),
    "karman2d.gif": (
        "outputs/karman_vortex_2d.mp4",
        0.0,
        5.0,
        10.0,
        720,
        256,
        0,
    ),
    "sac-forward.gif": (
        "outputs/sac_minimal/videos/eel2d_forward_policy.mp4",
        3.0,
        7.0,
        10.0,
        320,
        96,
        160,
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


def extract_gif(
    video_path: Path,
    start_seconds: float,
    duration_seconds: float,
    output_fps: float,
    output_width: int,
    colors: int,
    crop_bottom: int,
    output_path: Path,
) -> None:
    """Write a compact, optionally bottom-cropped GIF from a demo video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or source_frames <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid timing metadata: {video_path}")

    start_frame = round(start_seconds * source_fps)
    output_frames = max(1, round(duration_seconds * output_fps))
    frames: list[Image.Image] = []
    for output_index in range(output_frames):
        source_index = start_frame + round(output_index * source_fps / output_fps)
        if source_index >= source_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not read frame {source_index} from {video_path}")

        height, width = frame.shape[:2]
        output_height = round(height * output_width / width)
        resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        if crop_bottom:
            resized = resized[:-crop_bottom, :]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    capture.release()

    if not frames:
        raise RuntimeError(f"No GIF frames extracted from {video_path}")

    thumbnail_width = min(160, output_width)
    thumbnail_height = round(frames[0].height * thumbnail_width / frames[0].width)
    palette_columns = 8
    palette_rows = (len(frames) + palette_columns - 1) // palette_columns
    palette_samples = Image.new(
        "RGB",
        (thumbnail_width * palette_columns, thumbnail_height * palette_rows),
    )
    for frame_index, frame in enumerate(frames):
        thumbnail = frame.resize(
            (thumbnail_width, thumbnail_height),
            resample=Image.Resampling.LANCZOS,
        )
        x = (frame_index % palette_columns) * thumbnail_width
        y = (frame_index // palette_columns) * thumbnail_height
        palette_samples.paste(thumbnail, (x, y))
    palette = palette_samples.convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=colors,
    )
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / output_fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"{video_path.relative_to(PROJECT_ROOT)} -> {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Regenerate screenshots and GIFs used by the README and MkDocs site."""
    output_dir = PROJECT_ROOT / "docs" / "assets" / "demos"
    for output_name, (video_name, fraction, crop_top) in SCREENSHOTS.items():
        extract_frame(
            PROJECT_ROOT / video_name,
            fraction,
            crop_top,
            output_dir / output_name,
        )
    for output_name, animation in ANIMATIONS.items():
        (
            video_name,
            start_seconds,
            duration_seconds,
            output_fps,
            output_width,
            colors,
            crop_bottom,
        ) = animation
        extract_gif(
            PROJECT_ROOT / video_name,
            start_seconds,
            duration_seconds,
            output_fps,
            output_width,
            colors,
            crop_bottom,
            output_dir / output_name,
        )


if __name__ == "__main__":
    main()
