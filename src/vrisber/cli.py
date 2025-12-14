"""
Command-line interface for Vrisber.

Provides the `vrisber` console entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .processor import VRSBSProcessor


def main() -> int:
    """Main entry point for the vrisber CLI."""
    parser = argparse.ArgumentParser(
        prog="vrisber",
        description=(
            "Vrisber - VR SBS Video Processor: "
            "Split, process, and remux VR Side-by-Side videos using imageProcessor-cli."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file with default config.json
  vrisber input.mp4

  # Process with custom output
  vrisber input.mp4 output.mp4

  # Process folder (batch)
  vrisber ./videos/

  # Process folder with explicit output directory
  vrisber ./videos/ --output-dir ./restored/

  # Test mode (skip imageProcessor processing; validates split/mux pipeline)
  vrisber input.mp4 --test-mode

  # Commands-only mode (generate PowerShell script without executing)
  vrisber input.mp4 --commands-only

  # Use custom config file
  vrisber input.mp4 --config custom_config.json

  # Override device or codec
  vrisber input.mp4 --device cuda:1 --codec hevc
        """,
    )

    parser.add_argument(
        "input",
        help="Input video file or directory",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output video file (for single-file mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (for folder mode)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config file (default: config.json)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: skip imageProcessor processing (split/mux only)",
    )
    parser.add_argument(
        "--commands-only",
        action="store_true",
        help="Commands-only mode: generate PowerShell script without executing",
    )
    parser.add_argument(
        "--device",
        help="Override device (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--codec",
        help="Override final output codec (e.g., hevc_nvenc, hevc, libsvtav1, h264)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Force enable parallel eye processing",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Force disable parallel eye processing",
    )

    args = parser.parse_args()

    # Load processor
    try:
        processor = VRSBSProcessor(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Apply command-line overrides to config in memory
    if args.test_mode:
        processor.config.setdefault("processing", {})["test_mode"] = True
        processor.logger.info("TEST MODE ENABLED - imageProcessor processing will be skipped")

    if args.commands_only:
        processor.config.setdefault("processing", {})["commands_only"] = True
        processor.logger.info("COMMANDS-ONLY MODE ENABLED - PowerShell script will be generated without execution")

    if args.device:
        processor.config.setdefault("imageProcessor", {})["device"] = args.device
        processor.logger.info("Overriding device: %s", args.device)

    if args.codec:
        processor.config.setdefault("encoding", {})["codec"] = args.codec
        processor.logger.info("Overriding final output codec: %s", args.codec)

    if args.parallel and args.no_parallel:
        processor.logger.warning(
            "Both --parallel and --no-parallel specified; using configuration value."
        )
    elif args.parallel:
        processor.config.setdefault("processing", {})["parallel_eyes"] = True
        processor.logger.info("Forcing parallel eye processing: enabled")
    elif args.no_parallel:
        processor.config.setdefault("processing", {})["parallel_eyes"] = False
        processor.logger.info("Forcing parallel eye processing: disabled")

    # Check external dependencies (ffmpeg, ffprobe, imageProcessor-cli)
    # Skip dependency check in commands-only mode since we're just generating scripts
    if not args.commands_only:
        try:
            processor._check_dependencies()
        except RuntimeError as e:
            processor.logger.error(str(e))
            return 1

    input_path = Path(args.input)

    if input_path.is_file():
        # Single-file mode
        success = processor.process_file(str(input_path), args.output)
        return 0 if success else 1

    if input_path.is_dir():
        # Folder (batch) mode
        results = processor.process_folder(str(input_path), args.output_dir)
        return 0 if not results["failed"] else 1

    processor.logger.error("Input not found: %s", input_path)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())