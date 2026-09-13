"""Command-line interface for annotation and inference."""

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .io import list_images, load_image, save_hint
from .pipeline import ShorelinePipeline


def _default_project_root() -> Path:
    """Find the repository when invoked from it or from its parent folder."""
    current = Path.cwd().resolve()
    editable_install_root = Path(__file__).resolve().parents[2]
    candidates = (
        current,
        current / "yam-yabasha",
        editable_install_root,
    )
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yam-yabasha",
        description="Extract shorelines from fixed-camera images.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Folder containing data/ and outputs/ (default: auto-detected).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser(
        "annotate",
        help="Draw a coarse shoreline search hint on a location reference.",
    )
    annotate.add_argument("--location", required=True)
    annotate.add_argument(
        "--reference",
        type=Path,
        help="Override data/<location>/reference.jpg.",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--location", required=True)
    common.add_argument("--device", choices=["cpu", "cuda"])
    common.add_argument("--corridor-radius", type=float)
    common.add_argument("--contrast-threshold", type=float)
    common.add_argument("--water-prompt", action="append")
    common.add_argument("--land-prompt", action="append")
    common.add_argument("--no-registration", action="store_true")

    run = subparsers.add_parser("run", parents=[common], help="Process one image.")
    run.add_argument("--image", type=Path, required=True)

    batch = subparsers.add_parser(
        "batch",
        parents=[common],
        help="Process all supported images in a folder.",
    )
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--max-images", type=int)

    workflow = subparsers.add_parser(
        "workflow",
        help="Annotate camera references and process all configured locations.",
    )
    workflow.add_argument(
        "--locations",
        nargs="+",
        default=["entrance", "porch"],
        help="Locations to process in order (default: entrance porch).",
    )
    workflow.add_argument(
        "--input-root",
        type=Path,
        default=Path("Test"),
        help="Folder containing one image folder per location.",
    )
    workflow.add_argument(
        "--max-images",
        type=int,
        help="Maximum images to process per location.",
    )
    workflow.add_argument("--device", choices=["cpu", "cuda"])
    workflow.add_argument("--corridor-radius", type=float)
    workflow.add_argument("--contrast-threshold", type=float)
    workflow.add_argument("--water-prompt", action="append")
    workflow.add_argument("--land-prompt", action="append")
    workflow.add_argument("--no-registration", action="store_true")
    hint_mode = workflow.add_mutually_exclusive_group()
    hint_mode.add_argument(
        "--redraw-hints",
        action="store_true",
        help="Open every reference and replace all saved hints.",
    )
    hint_mode.add_argument(
        "--reuse-hints",
        action="store_true",
        help="Reuse saved hints without asking; annotate only missing hints.",
    )
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(project_root=args.project_root.resolve())
    segmentation = config.segmentation
    contour = config.contour
    if args.device:
        segmentation = replace(segmentation, device=args.device)
    if args.water_prompt:
        segmentation = replace(segmentation, water_prompts=tuple(args.water_prompt))
    if args.land_prompt:
        segmentation = replace(segmentation, land_prompts=tuple(args.land_prompt))
    if args.corridor_radius is not None:
        contour = replace(contour, corridor_radius_px=args.corridor_radius)
    if args.contrast_threshold is not None:
        contour = replace(contour, contrast_threshold=args.contrast_threshold)
    return replace(config, segmentation=segmentation, contour=contour)


def _annotate_location(
    project_root: Path,
    location: str,
    reference_path: Path | None = None,
) -> int:
    reference_path = reference_path or (
        project_root / "data" / location / "reference.jpg"
    )
    from matplotlib.backend_bases import MouseButton
    import matplotlib.pyplot as plt

    reference = load_image(reference_path)
    reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(reference_rgb)
    axis.set_title(
        f"{location}: left click adds, right click undoes, "
        "Enter or middle click saves"
    )
    axis.axis("off")
    figure.tight_layout()
    print(
        "Left click: add point | Right click: undo | "
        "Enter or middle click: save"
    )
    try:
        points = plt.ginput(
            n=-1,
            timeout=0,
            show_clicks=True,
            mouse_add=MouseButton.LEFT,
            mouse_pop=MouseButton.RIGHT,
            mouse_stop=MouseButton.MIDDLE,
        )
    finally:
        plt.close(figure)

    points_array = np.asarray(points, dtype=np.float64)
    if len(points_array) < 2:
        print("No hint saved: the window was closed or fewer than two points were selected.")
        return 1

    hint_path = project_root / "data" / location / "shoreline_hint.json"
    save_hint(hint_path, points_array)
    print(f"Saved {len(points_array)} hint points to {hint_path}")
    return 0


def _annotate(args: argparse.Namespace) -> int:
    return _annotate_location(
        args.project_root.resolve(),
        args.location,
        args.reference,
    )


def _should_redraw_hint(
    hint_path: Path,
    *,
    redraw_all: bool,
    reuse_all: bool,
) -> bool:
    if not hint_path.exists():
        return True
    if redraw_all:
        return True
    if reuse_all:
        return False
    try:
        answer = input(
            f"A saved hint exists at {hint_path}. Redraw it? [y/N]: "
        ).strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def _workflow(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    print("\nStep 1/2: prepare one reference hint per location.")
    for location in args.locations:
        hint_path = project_root / "data" / location / "shoreline_hint.json"
        redraw = _should_redraw_hint(
            hint_path,
            redraw_all=args.redraw_hints,
            reuse_all=args.reuse_hints,
        )
        if redraw:
            print(f"\nOpening the {location} reference...")
            if _annotate_location(project_root, location) != 0:
                print("Workflow stopped before image processing.")
                return 1
        else:
            print(f"Reusing saved {location} hint: {hint_path}")

    print("\nStep 2/2: register and process all location images.")
    pipeline = ShorelinePipeline(_config(args))
    register = not args.no_registration
    input_root = (
        args.input_root
        if args.input_root.is_absolute()
        else project_root / args.input_root
    )
    total_succeeded = 0
    failures: list[tuple[str, Path, str]] = []
    for location in args.locations:
        images = list_images(input_root / location)
        if args.max_images is not None:
            images = images[: args.max_images]
        if not images:
            print(f"\n{location}: no images found in {input_root / location}")
            continue

        succeeded = 0
        for image_path in tqdm(images, desc=f"Shoreline {location}"):
            try:
                pipeline.process(
                    image_path,
                    location,
                    register=register,
                )
                succeeded += 1
            except Exception as error:
                failures.append((location, image_path, str(error)))
        total_succeeded += succeeded
        print(
            f"{location}: {succeeded}/{len(images)} images completed. "
            f"Outputs: {pipeline.config.resolved_output_dir / location}"
        )

    if failures:
        print(f"\nCompleted with {len(failures)} failed images:")
        for location, image_path, error in failures:
            print(f"- [{location}] {image_path.name}: {error}")
    else:
        print(f"\nWorkflow complete: {total_succeeded} images processed successfully.")
    return 1 if failures else 0


def _print_result(result) -> None:
    registration = (
        result.registration.metrics
        if result.registration is not None
        else {"registration_skipped": True}
    )
    print(f"Image: {result.image_path}")
    print(f"Registration: {registration}")
    print(
        f"Shoreline: {len(result.shoreline.points)} points, "
        f"{len(result.shoreline.candidates)} candidates, "
        f"confidence={result.shoreline.confidence:.3f}"
    )
    print(f"Outputs: {result.output_paths['figure'].parent}")


def main(argv: list[str] | None = None) -> int:
    """Run the command-line application."""
    args = _parser().parse_args(argv)
    if args.command == "annotate":
        return _annotate(args)
    if args.command == "workflow":
        return _workflow(args)

    pipeline = ShorelinePipeline(_config(args))
    register = not args.no_registration
    if args.command == "run":
        result = pipeline.process(
            args.image,
            args.location,
            register=register,
        )
        _print_result(result)
        return 0

    results = pipeline.process_folder(
        args.input,
        args.location,
        register=register,
        max_images=args.max_images,
    )
    for result in results:
        _print_result(result)
    print(f"Processed {len(results)} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
