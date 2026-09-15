from __future__ import annotations

import argparse

from .pipeline import run_pipeline
from .stages.assemble import run_assembly
from .stages.segment import run_segmentation
from .stages.synthesize import run_synthesis
from .stages.transform import run_transformation
from .stages.transcribe import run_transcription
from .storage import derive_project_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relecture", description="Audience-adaptive lecture re-narration pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_project_argument(subparser: argparse.ArgumentParser, required: bool = True) -> None:
        subparser.add_argument("--project", required=required, help="Path to the project manifest JSON file.")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline.")
    add_project_argument(run_parser, required=False)
    run_parser.add_argument("--source-video", help="Source lecture video used when creating a new project.")
    run_parser.add_argument("--project-name", help="Optional project name when creating a new project.")
    run_parser.add_argument("--background", default="none", choices=["none", "cs_background", "custom"])
    run_parser.add_argument("--accessibility", default="none", choices=["none", "visual_impairment"])
    run_parser.add_argument("--describe-visuals", action="store_true", help="Describe slide visuals for visually impaired learners.")
    run_parser.add_argument("--target-audience", default="general audience")
    run_parser.add_argument("--custom-instructions")
    run_parser.add_argument(
        "--strategy",
        default="independent",
        choices=["independent", "full_context", "two_pass", "sliding_window"],
    )
    run_parser.add_argument("--mode", default="full", choices=["preview", "full"])
    run_parser.add_argument("--voice-sample-path")
    run_parser.add_argument("--language", default="en")
    run_parser.add_argument("--style", default="neutral")
    run_parser.add_argument("--model", default="qwen3")
    run_parser.add_argument("--backend", default="qwen3", choices=["qwen3", "chatterbox", "cosyvoice"])
    run_parser.add_argument("--output-suffix")
    run_parser.add_argument("--speed", type=float, default=1.0)
    run_parser.add_argument("--temperature", type=float, default=0.9)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--repetition-penalty", type=float, default=1.05)
    run_parser.add_argument("--output-filename", default="lecture_final.mp4")
    run_parser.add_argument("--no-subtitles", action="store_true")

    segment_parser = subparsers.add_parser("segment", help="Run segmentation.")
    add_project_argument(segment_parser, required=False)
    segment_parser.add_argument("--source-video")
    segment_parser.add_argument("--project-name")

    transcribe_parser = subparsers.add_parser("transcribe", help="Run transcription.")
    add_project_argument(transcribe_parser)
    transcribe_parser.add_argument("--language", default="en")

    transform_parser = subparsers.add_parser("transform", help="Run transcript transformation.")
    add_project_argument(transform_parser)
    transform_parser.add_argument("--segments", help="Comma-separated segment IDs or ranges, e.g. 1,2,4-6")
    transform_parser.add_argument("--background", default="none", choices=["none", "cs_background", "custom"])
    transform_parser.add_argument("--accessibility", default="none", choices=["none", "visual_impairment"])
    transform_parser.add_argument("--describe-visuals", action="store_true", help="Describe slide visuals for visually impaired learners.")
    transform_parser.add_argument("--target-audience", default="general audience")
    transform_parser.add_argument("--custom-instructions")
    transform_parser.add_argument(
        "--strategy",
        default="independent",
        choices=["independent", "full_context", "two_pass", "sliding_window"],
    )

    synthesize_parser = subparsers.add_parser("synthesize", help="Run synthesis.")
    add_project_argument(synthesize_parser)
    synthesize_parser.add_argument("--segments", help="Comma-separated segment IDs or ranges.")
    synthesize_parser.add_argument("--mode", default="preview", choices=["preview", "full"])
    synthesize_parser.add_argument("--voice-sample-path")
    synthesize_parser.add_argument("--language", default="en")
    synthesize_parser.add_argument("--style", default="neutral")
    synthesize_parser.add_argument("--model", default="qwen3")
    synthesize_parser.add_argument("--backend", default="qwen3", choices=["qwen3", "chatterbox", "cosyvoice"])
    synthesize_parser.add_argument("--output-suffix")
    synthesize_parser.add_argument("--speed", type=float, default=1.0)
    synthesize_parser.add_argument("--temperature", type=float, default=0.9)
    synthesize_parser.add_argument("--top-p", type=float, default=1.0)
    synthesize_parser.add_argument("--repetition-penalty", type=float, default=1.05)

    subparsers.add_parser("ui", help="Launch the Gradio demo UI.")

    assemble_parser = subparsers.add_parser("assemble", help="Assemble the final lecture video.")
    add_project_argument(assemble_parser)
    assemble_parser.add_argument("--segments", help="Comma-separated segment IDs or ranges.")
    assemble_parser.add_argument("--output-filename", default="lecture_final.mp4")
    assemble_parser.add_argument("--no-subtitles", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation metrics.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_transcription = eval_sub.add_parser("transcription", help="Compute WER against VTT ground truth.")
    eval_transcription.add_argument("--project", required=True)
    eval_transcription.add_argument("--ground-truth", required=True)

    eval_transformation = eval_sub.add_parser("transformation", help="LLM-as-judge transformation quality.")
    eval_transformation.add_argument("--project", required=True)

    eval_vision = eval_sub.add_parser("vision", help="LLM-as-judge vision description quality.")
    eval_vision.add_argument("--project", required=True)
    eval_vision.add_argument("--judge-model", help="Optional model name for the LLM judge.")

    eval_synthesis = eval_sub.add_parser("synthesis", help="Speaker similarity evaluation.")
    eval_synthesis.add_argument("--project", required=True)

    eval_aggregate = eval_sub.add_parser("aggregate", help="Aggregate eval results across dataset.")
    eval_aggregate.add_argument("--dataset", required=True)
    eval_aggregate.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.project and not args.source_video:
            parser.error("run requires --project or --source-video")
        project_file = derive_project_file(args.source_video, args.project) if args.source_video else args.project
        run_pipeline(
            project_file,
            source_video=args.source_video,
            project_name=args.project_name,
            background_profile=args.background,
            accessibility_profile=args.accessibility,
            target_audience=args.target_audience,
            describe_visuals=args.describe_visuals or (args.accessibility == "visual_impairment"),
            custom_instructions=args.custom_instructions,
            strategy=args.strategy,
            synthesis_mode=args.mode,
            voice_sample_path=args.voice_sample_path,
            language=args.language,
            style=args.style,
            model=args.model,
            backend=args.backend,
            output_suffix=args.output_suffix,
            speed=args.speed,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            output_filename=args.output_filename,
            generate_subtitles=not args.no_subtitles,
        )
        return 0

    if args.command == "segment":
        if not args.project and not args.source_video:
            parser.error("segment requires --project or --source-video")
        project_file = derive_project_file(args.source_video, args.project) if args.source_video else args.project
        run_segmentation(project_file, source_video=args.source_video, project_name=args.project_name)
        return 0

    if args.command == "transcribe":
        run_transcription(args.project, language=args.language)
        return 0

    if args.command == "transform":
        run_transformation(
            args.project,
            segment_ids=args.segments,
            background_profile=args.background,
            accessibility_profile=args.accessibility,
            target_audience=args.target_audience,
            describe_visuals=args.describe_visuals or (args.accessibility == "visual_impairment"),
            custom_instructions=args.custom_instructions,
            strategy=args.strategy,
        )
        return 0

    if args.command == "synthesize":
        run_synthesis(
            args.project,
            segment_ids=args.segments,
            mode=args.mode,
            voice_sample_path=args.voice_sample_path,
            language=args.language,
            style=args.style,
            model=args.model,
            backend=args.backend,
            output_suffix=args.output_suffix,
            speed=args.speed,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        return 0

    if args.command == "assemble":
        run_assembly(
            args.project,
            segment_ids=args.segments,
            output_filename=args.output_filename,
            generate_subtitles=not args.no_subtitles,
        )
        return 0

    if args.command == "eval":
        if args.eval_command == "transcription":
            from .eval.transcription import run_transcription_eval
            run_transcription_eval(args.project, args.ground_truth)
            return 0
        if args.eval_command == "transformation":
            from .eval.transformation import run_transformation_eval
            run_transformation_eval(args.project)
            return 0
        if args.eval_command == "vision":
            from .eval.vision import run_vision_eval
            run_vision_eval(args.project, judge_model=args.judge_model)
            return 0
        if args.eval_command == "synthesis":
            from .eval.synthesis import run_synthesis_eval
            run_synthesis_eval(args.project)
            return 0
        if args.eval_command == "aggregate":
            from .eval.aggregate import run_aggregate
            run_aggregate(args.dataset, args.output)
            return 0

    if args.command == "ui":
        from .ui import main as ui_main
        ui_main()
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
