"""Assemble prepared image frames into an MP4 video using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from PIL import Image


def check_ffmpeg() -> str:
    """Return the path to ffmpeg, or raise if not found."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise SystemExit(
            "ffmpeg not found. Install it with: brew install ffmpeg"
        )
    return path


def compile_video(
    frames: Iterable[Image.Image],
    output: Path,
    *,
    fps: int,
    width: int,
    height: int,
    total: int | None = None,
) -> None:
    """Stream RGB frames to ffmpeg and produce an H.264 MP4.

    Args:
        frames: Iterable of PIL Images, already sized to width x height.
        output: Path to write the .mp4 file.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        total: Optional frame count (for progress reporting upstream).
    """
    ffmpeg = check_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",  # overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # read from stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",  # compatibility
        "-preset", "medium",
        "-crf", "18",  # high quality
        "-movflags", "+faststart",  # web-friendly
        str(output),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass

    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{stderr}")


def compile_video_variable(
    frames: list[Image.Image],
    durations: list[float],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int = 30,
) -> None:
    """Encode frames with per-frame hold durations using the concat demuxer.

    Each `frames[i]` is held for `durations[i]` seconds. Output is a constant-
    fps H.264 MP4 (the variable timing comes from the concat input).
    """
    import tempfile

    if len(frames) != len(durations):
        raise ValueError(
            f"frames ({len(frames)}) and durations ({len(durations)}) length mismatch"
        )
    if not frames:
        raise ValueError("compile_video_variable: no frames to encode")

    ffmpeg = check_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="sfa_concat_") as tmpdir:
        tmp = Path(tmpdir)
        # Write frames as PNG (lossless; libx264 will compress).
        frame_paths: list[Path] = []
        for i, frame in enumerate(frames):
            p = tmp / f"f_{i:06d}.png"
            frame.save(p, format="PNG")
            frame_paths.append(p)

        # Build the concat list. The demuxer ignores the last duration, so
        # we duplicate the final frame entry without a duration to honor it.
        list_path = tmp / "list.txt"
        with open(list_path, "w") as f:
            for p, d in zip(frame_paths, durations):
                f.write(f"file '{p.as_posix()}'\n")
                f.write(f"duration {d:.6f}\n")
            f.write(f"file '{frame_paths[-1].as_posix()}'\n")

        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-fps_mode", "vfr",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "18",
            "-vf", f"scale={width}:{height}",
            "-movflags", "+faststart",
            str(output),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg variable-duration encode failed:\n{result.stderr}"
            )


def get_duration(path: Path) -> float:
    """Get the duration of a media file in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SystemExit("ffprobe not found. Install it with: brew install ffmpeg")

    result = subprocess.run(
        [
            ffprobe,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())


def mux_audio(
    video: Path,
    audio: Path,
    output: Path,
    *,
    audio_fade_out: float = 2.0,
) -> None:
    """Combine a video file with an audio track.

    Trims or loops audio to match video length, with a fade-out at the end.

    Args:
        video: Path to the silent video file.
        audio: Path to the audio file (mp3, m4a, wav, etc.).
        output: Path to write the final muxed video.
        audio_fade_out: Duration of audio fade-out in seconds.
    """
    ffmpeg = check_ffmpeg()

    video_duration = get_duration(video)
    audio_duration = get_duration(audio)

    # Build audio filter: loop if needed, trim to video length, fade out
    filters = []

    if audio_duration < video_duration:
        # Loop audio to cover video length
        loop_count = int(video_duration / audio_duration) + 1
        filters.append(f"aloop=loop={loop_count}:size=2e+09")

    fade_start = max(0, video_duration - audio_fade_out)
    filters.append(f"atrim=0:{video_duration}")
    filters.append(f"afade=t=out:st={fade_start}:d={audio_fade_out}")

    af = ",".join(filters)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video),
        "-i", str(audio),
        # Explicit stream mapping: some music sources (e.g. .webm) carry a
        # video track, which ffmpeg's default "best stream" picker would
        # otherwise prefer over our H.264 output, producing a file that
        # QuickTime refuses to open.
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-af", af,
        "-shortest",
        "-movflags", "+faststart",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed:\n{result.stderr}")
