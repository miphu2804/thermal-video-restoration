"""Browser demo server: upload a thermal video or pick a FLIR sequence, get A/B/C.

Each request runs ``scripts/infer_realtime.py`` headless, so the server shows
exactly what the CLI demo computes, then re-encodes the result to H.264 because
browsers cannot play OpenCV's mp4v output. Requires ffmpeg on PATH.

    uv run --extra detector --extra demo python scripts/demo_server.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = list(
    json.loads((ROOT / "configs/abc_baselines.json").read_text())["pipelines"]
)
STEPS = ["contrast", "blur", "fixed_stripes", "temporal_stripes", "noise"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt",
    )
    parser.add_argument(
        "--flir-root",
        type=Path,
        default=ROOT / "data/FLIR_ADAS_v2/video_thermal_test",
        help="TIFF directory whose sequences appear in the dropdown.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data/test_inference",
        help="Directory of local .mp4 files offered as one-click sample inputs.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Public Gradio link.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    sequences = _list_sequences(args.flir_root)

    def run(video, sequence, mode, pipeline, level, steps, seed, threshold, frames):
        return _run_demo(
            args, video, sequence, mode, pipeline, level, steps, seed, threshold, frames
        )

    with gr.Blocks(title="Thermal Video Restoration") as app:
        gr.Markdown(
            "# Thermal Video Restoration demo\n"
            "**A** source → **B** simulated degradation → **C** restoration, "
            "all through the same frozen detector. Uploaded videos are treated "
            "as 8-bit AGC thermal (not raw radiometric); there is no mAP without labels."
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload thermal video (takes priority)")
                sequence = gr.Dropdown(
                    sequences, label=f"…or FLIR sequence ({args.flir_root.name})"
                )
                mode = gr.Radio(
                    ["A/B/C comparison", "Detection only"],
                    value="A/B/C comparison",
                    label="Mode",
                )
                pipeline = gr.Dropdown(
                    PIPELINES, value=PIPELINES[-1], label="Restoration (C)"
                )
                level = gr.Radio(
                    ["light", "medium", "heavy"], value="medium", label="Level"
                )
                steps = gr.CheckboxGroup(
                    STEPS, value=STEPS, label="Degradation steps (B)"
                )
                with gr.Row():
                    seed = gr.Number(0, precision=0, minimum=0, label="Seed")
                    threshold = gr.Slider(0, 1, 0.3, step=0.05, label="Score")
                    frames = gr.Slider(1, 900, 150, step=1, label="Max frames")
                samples = sorted(args.samples.glob("*.mp4"))
                if samples:
                    gr.Examples(
                        [[str(path)] for path in samples],
                        [video],
                        label=f"Sample videos ({args.samples.name})",
                    )
                button = gr.Button("Run", variant="primary")
            with gr.Column():
                output = gr.Video(label="Result")
                log = gr.Textbox(label="Log", lines=12)
        button.click(
            run,
            [video, sequence, mode, pipeline, level, steps, seed, threshold, frames],
            [output, log],
        )
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


def _list_sequences(root: Path) -> list[str]:
    """Return FLIR sequence names, or none when the dataset is not downloaded."""
    if not root.is_dir():
        return []
    from thermal_restore.data.experiment_config import source_sequence

    names = {source_sequence(path) for path in root.rglob("*.tif*")}
    return sorted(name for name in names if name.startswith("video-"))


def _run_demo(
    args, video, sequence, mode, pipeline, level, steps, seed, threshold, frames
):
    if not video and not sequence:
        raise gr.Error("Upload a video or choose a FLIR sequence.")
    if mode == "A/B/C comparison" and not steps:
        raise gr.Error("Select at least one degradation step.")
    out_dir = ROOT / "outputs/demo_server" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    raw_video = out_dir / "raw.mp4"
    command = [
        sys.executable,
        str(ROOT / "scripts/infer_realtime.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--no-display",
        "--target-fps",
        "0",
        "--max-frames",
        str(int(frames)),
        "--score-threshold",
        str(threshold),
        "--output-video",
        str(raw_video),
    ]
    if video:
        command += ["--source", video, "--allow-uint8"]
    else:
        command += ["--source", str(args.flir_root), "--sequence", sequence]
    if mode == "A/B/C comparison":
        command += [
            "--compare-abc",
            "--pipeline",
            pipeline,
            "--level",
            level,
            "--seed",
            str(int(seed)),
            "--degradation-steps",
            *steps,
        ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    log = result.stdout + result.stderr
    if result.returncode != 0 or not raw_video.is_file():
        raise gr.Error(f"Inference failed:\n{log[-2000:]}")
    web_video = out_dir / "result.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video)]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(web_video)],
        check=True,
    )
    return str(web_video), log


if __name__ == "__main__":
    main()
