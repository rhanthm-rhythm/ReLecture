from __future__ import annotations

"""Guided Gradio demo for the ReLecture pipeline."""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import gradio as gr
import requests

from .config import load_config
from .logger import setup_run_logger, get_run_logger
from .media import check_ffmpeg_available
from .services import TranscriptionService
from .stages.assemble import run_assembly
from .stages.segment import run_segmentation
from .stages.synthesize import build_tts_backend, run_synthesis
from .stages.transform import TRANSFORMATION_STRATEGIES, run_transformation
from .stages.transcribe import run_transcription
from .storage import project_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = get_run_logger("relecture.ui")

STAGES = ("Preflight", "Segmentation", "Transcription", "Transformation", "Synthesis", "Assembly")


def _uploaded_path(upload) -> str | None:
    if upload is None:
        return None
    return upload if isinstance(upload, str) else upload.name


def _default_output_root() -> Path:
    """Root directory under which per-run folders are created.

    Overridable via RELECTURE_OUTPUT_ROOT so tests (and anyone who wants runs
    written elsewhere) don't litter the real repo's output/ directory with
    throwaway run folders.
    """
    override = os.getenv("RELECTURE_OUTPUT_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "output"


def initialize_run(video_file, voice_file=None) -> dict[str, str | None]:
    video_path = _uploaded_path(video_file)
    if not video_path:
        raise ValueError("Upload a lecture video before starting the pipeline.")

    run_dir = _default_output_root() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logger(run_dir)
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True)
    source_copy = input_dir / Path(video_path).name
    shutil.copy2(video_path, source_copy)

    voice_copy = None
    voice_path = _uploaded_path(voice_file)
    if voice_path:
        voice_copy = input_dir / Path(voice_path).name
        if voice_copy == source_copy:
            voice_copy = input_dir / f"voice_{Path(voice_path).name}"
        shutil.copy2(voice_path, voice_copy)

    return {
        "run_id": run_dir.name,
        "project_file": str(run_dir / "project.json"),
        "source_video": str(source_copy),
        "voice_sample": str(voice_copy) if voice_copy else None,
    }


def _port_healthy(port: int) -> bool:
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _light_html(ready: bool) -> str:
    color = "#22c55e" if ready else "#ef4444"
    glow = f"box-shadow: 0 0 8px 2px {color}80;" if ready else ""
    return (
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'border-radius:50%;background:{color};{glow}"></span>'
    )


def _health_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/health", "", ""))


def _endpoint_health(endpoint: str) -> bool:
    try:
        response = requests.get(_health_endpoint(endpoint), timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def collect_readiness(backend: str, accessibility: str) -> list[list[str]]:
    config = load_config()
    rows: list[list[str]] = []

    rows.append(["FFmpeg", "Ready" if check_ffmpeg_available() else "Unavailable", "Required for media processing"])

    whisper_ready = TranscriptionService(config.whisper_endpoint).health()
    rows.append([
        "Whisper",
        "Ready" if whisper_ready else "Unavailable",
        config.whisper_endpoint,
    ])

    llm_configured = bool(config.llm.base_url and config.llm.model_name)
    if config.llm.base_url and "localhost" not in config.llm.base_url and "127.0.0.1" not in config.llm.base_url:
        llm_configured = llm_configured and bool(config.llm.api_key)
    rows.append([
        "Transformation LLM",
        "Ready" if llm_configured else "Not configured",
        config.llm.model_name or "Set LLM_MODEL_NAME and the provider credentials",
    ])

    try:
        tts_ready = build_tts_backend(backend).health()
    except Exception:
        tts_ready = False
    rows.append([
        f"TTS ({backend})",
        "Ready" if tts_ready else "Unavailable",
        "Selected synthesis backend",
    ])

    if accessibility == "visual_impairment":
        vision_configured = bool(config.vision.base_url and config.vision.model_name)
        vision_ready = vision_configured and _endpoint_health(config.vision.base_url)
        rows.append([
            "Vision model",
            "Ready" if vision_ready else ("Unavailable" if vision_configured else "Not configured"),
            config.vision.base_url or "Set VISION_BASE_URL",
        ])
    else:
        rows.append(["Vision model", "Not required", "Enable visual-impairment adaptation to use it"])

    return rows


def readiness_for_ui(backend: str, accessibility: str) -> tuple[list[list[str]], str]:
    rows = collect_readiness(backend, accessibility)
    blocking = [name for name, status, _ in rows if status in {"Unavailable", "Not configured"}]
    if blocking:
        return rows, f"Not ready: {', '.join(blocking)}"
    return rows, "All required services are ready."


def _process_rows(states: dict[str, tuple[str, str, str]]) -> list[list[str]]:
    return [[stage, *states[stage]] for stage in STAGES]


def _initial_process_states() -> dict[str, tuple[str, str, str]]:
    return {stage: ("Waiting", "", "") for stage in STAGES}


def _comparison_rows(transcription, transformation) -> list[list]:
    original_by_id = {segment.id: segment for segment in transcription.segments}
    rows = []
    for adapted in transformation.segments:
        original = original_by_id.get(adapted.id, adapted)
        rows.append([
            adapted.id,
            f"{adapted.duration:.1f}s",
            original.clean_transcript or original.raw_transcript or "",
            adapted.transformed_transcript or adapted.clean_transcript or adapted.raw_transcript or "",
        ])
    return rows


def _artifact_path(project_file: str, artifact) -> str | None:
    if artifact is None:
        return None
    return str(Path(project_paths(project_file).project_dir) / artifact.path)


def _format_failure(stage: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    hints = {
        "Preflight": "Start the required local services or update their endpoint configuration.",
        "Segmentation": "Check that the uploaded video is readable and FFmpeg/OpenCV are installed.",
        "Transcription": "Check the Whisper server and transformation LLM configuration.",
        "Transformation": "Check the LLM configuration and, for visual accessibility, the vision service.",
        "Synthesis": "Check the selected TTS server and the voice-reference audio.",
        "Assembly": "Full synthesis must complete and FFmpeg must be available.",
    }
    return f"{stage} failed: {detail} {hints.get(stage, '')}".strip()


def run_demo_workflow(
    video_file,
    voice_file,
    mode,
    background,
    accessibility,
    audience,
    custom_instructions,
    strategy,
    backend,
    speed,
    temperature,
    top_p,
    repetition_penalty,
):
    import queue as _queue
    from concurrent.futures import ThreadPoolExecutor

    # Gradio's built-in gr.Progress()/tqdm bar rendering was unreliable in this
    # UI (visually broken, and having one tqdm tracker per stage created at
    # once made it worse) — instead we fold the percentage straight into the
    # status text shown in the "Pipeline progress" table below.
    _progress_pos = {stage: 0 for stage in STAGES}
    progress_events: "_queue.Queue[tuple[str, float, str]]" = _queue.Queue()

    def _cb(stage):
        def update(frac: float, desc: str):
            target = int(min(max(frac, 0.0), 1.0) * 100)
            _progress_pos[stage] = max(_progress_pos[stage], target)
            labeled_desc = f"[{target}%] {desc}"
            progress_events.put((stage, frac, labeled_desc))
            logger.info("[%s %3d%%] %s", stage, target, desc)
        return update

    states = _initial_process_states()
    comparison = []
    preview_audio = preview_video = final_video = None
    downloads: list[str] = []
    run_state = None
    started = time.perf_counter()

    def output(status: str, summary: str = ""):
        return (
            status,
            _process_rows(states),
            comparison,
            preview_audio,
            preview_video,
            final_video,
            downloads,
            summary,
            run_state,
        )

    def _run_stage_live(stage: str, runner, stage_started: float):
        """Run a stage function in a background thread, yielding live status
        updates as its progress_cb fires instead of blocking silently until
        the whole stage completes."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner)
            while True:
                try:
                    _, frac, desc = progress_events.get(timeout=0.3)
                except _queue.Empty:
                    if future.done():
                        break
                    continue
                states[stage] = ("Running", desc, f"{time.perf_counter() - stage_started:.1f}s")
                yield output(desc)
                if future.done():
                    break
            # Drain any events emitted right before completion.
            while True:
                try:
                    _, frac, desc = progress_events.get_nowait()
                except _queue.Empty:
                    break
                states[stage] = ("Running", desc, f"{time.perf_counter() - stage_started:.1f}s")
                yield output(desc)
            return future.result()  # re-raise any exception, else return the stage's manifest

    try:
        run_state = initialize_run(video_file, voice_file)
        logger.info(f"=== Starting new run {run_state['run_id']} ===")
        logger.info(f"Source video: {run_state['source_video']}, Voice sample: {run_state['voice_sample']}")
        logger.info(f"Settings: mode={mode}, backend={backend}, audience='{audience}', strategy={strategy}")
    except Exception as exc:
        logger.exception(f"Preflight initialization failed: {exc}")
        states["Preflight"] = ("Failed", str(exc), "")
        yield output(_format_failure("Preflight", exc))
        return

    states["Preflight"] = ("Running", "Checking required services", "")
    yield output("Checking service readiness…")
    preflight_started = time.perf_counter()
    readiness = collect_readiness(backend, accessibility)
    blockers = [name for name, status, _ in readiness if status in {"Unavailable", "Not configured"}]
    if blockers:
        detail = f"Required services are not ready: {', '.join(blockers)}"
        logger.error(f"Preflight check failed: {detail}")
        states["Preflight"] = ("Failed", detail, f"{time.perf_counter() - preflight_started:.1f}s")
        yield output(_format_failure("Preflight", RuntimeError(detail)))
        return
    states["Preflight"] = ("Completed", "All required services ready", f"{time.perf_counter() - preflight_started:.1f}s")
    logger.info("Preflight check passed: all required services ready.")
    yield output("Preflight complete. Starting segmentation…")

    project_file = run_state["project_file"]
    stage_specs = [
        ("Segmentation", lambda: run_segmentation(project_file, source_video=run_state["source_video"], progress_cb=_cb("Segmentation"))),
        ("Transcription", lambda: run_transcription(project_file, progress_cb=_cb("Transcription"))),
        (
            "Transformation",
            lambda: run_transformation(
                project_file,
                background_profile=background,
                accessibility_profile=accessibility,
                target_audience=audience,
                custom_instructions=custom_instructions or None,
                strategy=strategy,
                progress_cb=_cb("Transformation"),
            ),
        ),
        (
            "Synthesis",
            lambda: run_synthesis(
                project_file,
                mode=mode,
                voice_sample_path=run_state["voice_sample"],
                backend=backend,
                speed=float(speed),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                progress_cb=_cb("Synthesis"),
            ),
        ),
    ]

    manifests = {}
    for stage, runner in stage_specs:
        stage_started = time.perf_counter()
        states[stage] = ("Running", f"Running {stage.lower()}", "")
        logger.info("=== Stage %s: starting ===", stage)
        yield output(f"{stage} is running…")
        try:
            manifest = yield from _run_stage_live(stage, runner, stage_started)
        except Exception as exc:
            logger.exception("=== Stage %s: FAILED after %.1fs ===", stage, time.perf_counter() - stage_started)
            states[stage] = ("Failed", str(exc), f"{time.perf_counter() - stage_started:.1f}s")
            yield output(_format_failure(stage, exc))
            return
        logger.info("=== Stage %s: completed in %.1fs ===", stage, time.perf_counter() - stage_started)
        manifests[stage] = manifest
        elapsed = time.perf_counter() - stage_started
        if stage in {"Segmentation", "Transcription", "Transformation"}:
            result_detail = f"{len(manifest.segments)} segments"
        else:
            result_detail = f"{len(manifest.results)} result(s) with {manifest.backend}"
        if stage == "Transformation":
            comparison = _comparison_rows(manifests["Transcription"], manifest)
            if accessibility == "visual_impairment":
                result_detail += ", vision-assisted"
        logger.info(f"--- Completed stage {stage} in {elapsed:.1f}s: {result_detail} ---")
        states[stage] = ("Completed", result_detail, f"{elapsed:.1f}s")
        yield output(f"{stage} complete.")

    synthesis = manifests["Synthesis"]
    if synthesis.results:
        first_result = synthesis.results[0]
        preview_audio = _artifact_path(project_file, first_result.audio)
        preview_video = _artifact_path(project_file, first_result.video)

    if mode == "preview":
        states["Assembly"] = ("Skipped", "Preview mode does not assemble a final lecture", "")
    else:
        stage_started = time.perf_counter()
        states["Assembly"] = ("Running", "Combining synthesized segments", "")
        logger.info("=== Stage Assembly: starting ===")
        yield output("Assembly is running…")
        try:
            assembly = yield from _run_stage_live(
                "Assembly",
                lambda: run_assembly(project_file, generate_subtitles=True, progress_cb=_cb("Assembly")),
                stage_started,
            )
        except Exception as exc:
            logger.exception("=== Stage Assembly: FAILED after %.1fs ===", time.perf_counter() - stage_started)
            states["Assembly"] = ("Failed", str(exc), f"{time.perf_counter() - stage_started:.1f}s")
            yield output(_format_failure("Assembly", exc))
            return
        logger.info("=== Stage Assembly: completed in %.1fs ===", time.perf_counter() - stage_started)
        final_video = _artifact_path(project_file, assembly.output_video)
        subtitle = _artifact_path(project_file, assembly.subtitles)
        downloads = [path for path in [final_video, subtitle] if path and Path(path).exists()]
        states["Assembly"] = (
            "Completed",
            f"{len(assembly.included_segments)} segments, {assembly.duration_seconds:.1f}s",
            f"{time.perf_counter() - stage_started:.1f}s",
        )

    paths = project_paths(project_file)
    downloads.extend(
        str(path)
        for path in [
            Path(paths.project_file),
            Path(paths.project_dir) / "run.log",
            Path(paths.manifests_dir) / "transformation.v1.json",
            Path(paths.manifests_dir) / "synthesis.v1.json",
        ]
        if path.exists() and str(path) not in downloads
    )
    elapsed = time.perf_counter() - started
    summary = (
        f"### Run complete\n"
        f"**Mode:** {mode} · **Segments:** {len(manifests['Segmentation'].segments)} · "
        f"**Strategy:** {strategy} · **TTS:** {backend} · **Elapsed:** {elapsed:.1f}s\n\n"
        f"Project: `{project_file}`"
    )
    yield output("Preview complete." if mode == "preview" else "Full adapted lecture complete.", summary)


def _toggle_custom(profile: str):
    return gr.update(visible=profile == "custom")


def _reset_outputs():
    return "", [], [], None, None, None, [], "", None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ReLecture — Lecture Adaptation Demo") as demo:
        run_state = gr.State(None)
        gr.Markdown(
            "# ReLecture\n"
            "Upload a lecture, choose its audience, and follow the manifest-driven pipeline from input to adapted speech and video."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Input and adaptation")
                video_input = gr.Video(label="Lecture video")
                voice_upload = gr.Audio(label="Voice reference (optional)", type="filepath")
                mode = gr.Radio(["preview", "full"], value="preview", label="Run mode")
                with gr.Row():
                    background = gr.Dropdown(
                        ["none", "cs_background", "custom"],
                        value="cs_background",
                        label="Background profile",
                    )
                    accessibility = gr.Dropdown(
                        ["none", "visual_impairment"],
                        value="none",
                        label="Accessibility profile",
                    )
                audience = gr.Textbox(value="students without a CS background", label="Target audience")
                custom_instructions = gr.Textbox(label="Custom adaptation instructions", visible=False)
                with gr.Row():
                    strategy = gr.Dropdown(list(TRANSFORMATION_STRATEGIES), value="independent", label="Strategy")
                    backend = gr.Dropdown(["qwen3", "chatterbox", "cosyvoice"], value="qwen3", label="TTS backend")
                with gr.Accordion("Advanced synthesis settings", open=False):
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                    temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.05, label="Repetition penalty")
                run_button = gr.Button("Run adaptation", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## 2. Service readiness")
                with gr.Row(elem_classes=["readiness-lights"]):
                    light_components = {}
                    for name, port in [
                        ("Whisper", 5001),
                        ("Chatterbox", 5002),
                        ("CosyVoice", 5003),
                        ("Qwen3", 5000),
                    ]:
                        light_components[name] = gr.HTML(
                            f'<div style="display:flex;align-items:center;gap:6px">'
                            f"{_light_html(False)}<span>{name} :{port}</span></div>"
                        )
                readiness_timer = gr.Timer(2)
                for name, port in [("Whisper", 5001), ("Chatterbox", 5002), ("CosyVoice", 5003), ("Qwen3", 5000)]:
                    readiness_timer.tick(
                        fn=lambda _name=name, _port=port: (
                            f'<div style="display:flex;align-items:center;gap:6px">'
                            f"{_light_html(_port_healthy(_port))}<span>{_name} :{_port}</span></div>"
                        ),
                        outputs=light_components[name],
                    )
                readiness_status = gr.Markdown("Select the configuration, then check required services.")
                readiness_table = gr.Dataframe(
                    headers=["Dependency", "Status", "Details"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    label="Preflight",
                )
                readiness_button = gr.Button("Refresh readiness")
                gr.Markdown("## 3. Processing")
                workflow_status = gr.Textbox(label="Current status", interactive=False)
                process_table = gr.Dataframe(
                    headers=["Stage", "State", "Result", "Elapsed"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    label="Pipeline progress",
                )

        gr.Markdown("## 4. Results")
        run_summary = gr.Markdown()
        comparison_table = gr.Dataframe(
            headers=["Segment", "Duration", "Original transcript", "Adapted transcript"],
            datatype=["number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Before / after",
        )
        with gr.Row():
            preview_audio = gr.Audio(label="Adapted preview audio", type="filepath")
            preview_video = gr.Video(label="Adapted segment preview")
            final_video = gr.Video(label="Final adapted lecture")
        downloads = gr.File(label="Download outputs and manifests", file_count="multiple")

        background.change(_toggle_custom, inputs=[background], outputs=[custom_instructions])
        readiness_button.click(
            readiness_for_ui,
            inputs=[backend, accessibility],
            outputs=[readiness_table, readiness_status],
        )
        video_input.change(
            _reset_outputs,
            outputs=[workflow_status, process_table, comparison_table, preview_audio, preview_video, final_video, downloads, run_summary, run_state],
        )
        run_button.click(
            run_demo_workflow,
            inputs=[
                video_input,
                voice_upload,
                mode,
                background,
                accessibility,
                audience,
                custom_instructions,
                strategy,
                backend,
                speed,
                temperature,
                top_p,
                repetition_penalty,
            ],
            outputs=[
                workflow_status,
                process_table,
                comparison_table,
                preview_audio,
                preview_video,
                final_video,
                downloads,
                run_summary,
                run_state,
            ],
        )

    return demo.queue(default_concurrency_limit=1)


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
