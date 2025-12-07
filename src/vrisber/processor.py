
"""
Vrisber core processing module.

Splits VR Side-by-Side (SBS) videos into left/right eye streams, runs each
eye through an external restoration tool (imageProcessor-cli), and then reconstructs
a final SBS output tuned for VR playback.

Design goals:
- Lossless (or effectively lossless) intermediates
- Exactly one lossy encode in the pipeline: the final SBS mux step
- Simple, JSON-driven configuration
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class VRSBSProcessor:
    """Main processor class for VR SBS videos."""

    def __init__(self, config_path: str = "config.json") -> None:
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Configuration & logging
    # -------------------------------------------------------------------------
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get("logging", {}).get("level", "INFO").upper()
        log_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Remove any existing handlers to avoid duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        console_handler.setFormatter(logging.Formatter(log_format))

        # File handler
        file_handler = logging.FileHandler("vrisber.log", encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(logging.Formatter(log_format))

        logging.root.setLevel(getattr(logging, log_level, logging.INFO))
        logging.root.addHandler(console_handler)
        logging.root.addHandler(file_handler)

    # -------------------------------------------------------------------------
    # Dependency checks
    # -------------------------------------------------------------------------
    def _check_dependencies(self) -> None:
        """Check if required external tools are available."""
        self.logger.info("Checking dependencies...")

        # ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.logger.info("[OK] ffmpeg found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")

        # ffprobe
        try:
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
            self.logger.info("[OK] ffprobe found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffprobe not found. Please install ffmpeg and ensure it is on PATH.")

        # imageProcessor-cli (unless in test mode)
        if not self.config.get("processing", {}).get("test_mode", False):
            imageProcessor_path = self.config["imageProcessor"]["path"]

            # Absolute path?
            if os.path.isabs(imageProcessor_path):
                if not os.path.exists(imageProcessor_path):
                    raise RuntimeError(f"imageProcessor-cli not found at: {imageProcessor_path}")
                self.logger.info(f"[OK] imageProcessor-cli found at {imageProcessor_path}")
                return

            # Otherwise, assume it's on PATH and probe it
            try:
                # Prefer --version; fall back to --help; be generous with timeouts
                result = subprocess.run(
                    [imageProcessor_path, "--version"],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.logger.info("[OK] imageProcessor-cli found in PATH")
                    return

                result = subprocess.run(
                    [imageProcessor_path, "--help"],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.logger.info("[OK] imageProcessor-cli found in PATH")
                    return

                raise RuntimeError(f"imageProcessor-cli not responding: {imageProcessor_path}")
            except subprocess.TimeoutExpired:
                # imageProcessor can be slow on first run; treat timeout as "probably fine"
                self.logger.warning("imageProcessor-cli check timed out (this can be normal on first run)")
                self.logger.info("[OK] imageProcessor-cli assumed available in PATH")
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(f"imageProcessor-cli not found in PATH or at: {imageProcessor_path}")

    # -------------------------------------------------------------------------
    # ffprobe analysis helpers
    # -------------------------------------------------------------------------
    def analyze_video(self, video_path: str) -> Dict:
        """Analyze video with ffprobe to get width/height/fps/codec/pix_fmt."""
        self.logger.info("Analyzing video: %s", video_path)

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name,bit_rate,pix_fmt",
            "-of", "json",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        if "streams" not in info or not info["streams"]:
            raise RuntimeError("No video stream found in input file")

        stream = info["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec_name = stream.get("codec_name", "unknown")
        pix_fmt = stream.get("pix_fmt", "unknown")
        bit_rate = int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else 0

        # Parse avg_frame_rate like "30000/1001"
        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        num, den = avg_frame_rate.split("/")
        fps = float(num) / float(den) if den != "0" else 0.0

        self.logger.info(
            "Video info: %dx%d, %.3f fps, codec=%s, pix_fmt=%s, bitrate=%d",
            width, height, fps, codec_name, pix_fmt, bit_rate,
        )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "codec_name": codec_name,
            "pix_fmt": pix_fmt,
            "bit_rate": bit_rate,
        }

    # -------------------------------------------------------------------------
    # imageProcessor integration
    # -------------------------------------------------------------------------
    def _build_imageProcessor_command(
        self,
        input_file: str,
        output_file: str,
        video_info: Dict,
        device: Optional[str] = None,
    ) -> List[str]:
        """
        Build the imageProcessor CLI command.

        Design:
        - Always set universal args: --input/--output/--device/--codec.
        - Honor max_clip_length via --max-clip-length.
        - For *all other* keys under config["imageProcessor"], auto-generate
          CLI flags:
            key  -> --key-with-underscores-turned-into-dashes value
          Booleans:
            true  -> --flag        (no value)
            false -> skipped
        - Append extra_args as a final raw string if present.
        """
        image_cfg = self.config["imageProcessor"]
        processing_cfg = self.config.get("processing", {})

        image_path = image_cfg["path"]
        max_clip = int(image_cfg.get("max_clip_length", 0) or 0)

        # Device selection
        device = device or image_cfg.get("device", "cuda:0")

        # Intermediate codec for backend output
        intermediate_codec = processing_cfg.get("intermediate_codec", "ffv1")
        intermediate_codec = str(intermediate_codec).lower()

        cmd: List[str] = [
            image_path,
            "--input", input_file,
            "--output", output_file,
            "--device", device,
            "--codec", intermediate_codec,
        ]

        # Limit clip length if requested
        if max_clip > 0:
            cmd.extend(["--max-clip-length", str(max_clip)])

        # ------------------------------------------------------------------
        # Generic flags from config["imageProcessor"]
        # ------------------------------------------------------------------
        # Keys we treat as "internal" and NOT as CLI flags:
        special_keys = {"path", "device", "max_clip_length", "extra_args"}

        for key, value in image_cfg.items():
            if key in special_keys:
                continue
            if value is None or value is False:
                continue

            # Allow keys like "mosaic_detection_model_path"
            # or "mosaic-detection-model-path"
            if key.startswith("--"):
                flag = key
            else:
                flag = "--" + key.replace("_", "-")

            if isinstance(value, bool):
                # True -> bare flag, False already skipped
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        # Extra args (verbatim string, still supported as backdoor)
        extra_args = image_cfg.get("extra_args")
        if extra_args:
            # Naive split; fine as long as paths don't have spaces.
            cmd.extend(extra_args.split())

        return cmd

    # -------------------------------------------------------------------------
    # Encoder options for final SBS mux
    # -------------------------------------------------------------------------
    def _build_encoder_options(
        self,
        codec: str,
        codec_config: Dict,
        keyframe_interval: int,
        fps: float,
    ) -> str:
        """
        Build ffmpeg encoder options for the final SBS encode.

        Returns a single string (to be split with shlex.split) so that all
        quoting remains under our control.
        """
        fps_rounded = int(round(fps)) if fps > 0 else 0

        if codec == "hevc_nvenc":
            preset = codec_config.get("preset", "p7")
            rc = codec_config.get("rc", "constqp")
            qp = codec_config.get("qp", 18)
            rc_lookahead = codec_config.get("rc_lookahead", 32)
            spatial_aq = codec_config.get("spatial_aq", 1)
            temporal_aq = codec_config.get("temporal_aq", 1)
            aq_strength = codec_config.get("aq_strength", 8)
            bf = codec_config.get("bf", 3)
            b_ref_mode = codec_config.get("b_ref_mode", "middle")

            opts = (
                f"-c:v hevc_nvenc "
                f"-preset {preset} "
                f"-rc {rc} "
                f"-qp {qp} "
                f"-rc-lookahead {rc_lookahead} "
                f"-spatial-aq {spatial_aq} "
                f"-temporal-aq {temporal_aq} "
                f"-aq-strength {aq_strength} "
                f"-bf {bf} "
                f"-b_ref_mode {b_ref_mode} "
            )
            if keyframe_interval > 0 and fps_rounded > 0:
                opts += f"-g {keyframe_interval} -r {fps_rounded} "
            opts += "-pix_fmt yuv420p"

        elif codec == "hevc":
            crf = codec_config.get("crf", 15)
            preset = codec_config.get("preset", "slower")
            x265_params = codec_config.get("x265_params", "aq-mode=3:psy-rd=1.0")

            opts = (
                f"-c:v hevc "
                f"-crf {crf} "
                f"-preset {preset} "
                f'-x265-params "{x265_params}" '
            )
            if keyframe_interval > 0 and fps_rounded > 0:
                opts += f"-g {keyframe_interval} -r {fps_rounded} "
            opts += "-pix_fmt yuv420p"

        elif codec == "libsvtav1":
            crf = codec_config.get("crf", 20)
            preset = codec_config.get("preset", 4)

            opts = (
                f"-c:v libsvtav1 "
                f"-crf {crf} "
                f"-preset {preset} "
            )
            if keyframe_interval > 0 and fps_rounded > 0:
                opts += f"-g {keyframe_interval} -r {fps_rounded} "
            opts += "-pix_fmt yuv420p"

        elif codec == "h264":
            crf = codec_config.get("crf", 18)
            preset = codec_config.get("preset", "slow")

            opts = (
                f"-c:v h264 "
                f"-crf {crf} "
                f"-preset {preset} "
            )
            if keyframe_interval > 0 and fps_rounded > 0:
                opts += f"-g {keyframe_interval} -r {fps_rounded} "
            opts += "-pix_fmt yuv420p"

        else:
            # Fallback to libx264 with sane defaults
            self.logger.warning("Unknown codec '%s', falling back to libx264", codec)
            opts = "-c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p"
            if keyframe_interval > 0 and fps_rounded > 0:
                opts += f" -g {keyframe_interval} -r {fps_rounded}"

        return opts

    # -------------------------------------------------------------------------
    # Split & audio extraction
    # -------------------------------------------------------------------------
    def _get_intermediate_codec_args(self) -> List[str]:
        """
        Build ffmpeg codec arguments for intermediate files (split + imageProcessor I/O).

        This uses processing.intermediate_codec and is intentionally separate
        from the final encoding configuration.
        """
        processing_cfg = self.config.get("processing", {})
        codec = str(processing_cfg.get("intermediate_codec", "ffv1")).lower()

        if codec == "ffv1":
            # Matroska + FFV1 is a very robust lossless combo
            return ["-c:v", "ffv1", "-level", "3"]
        else:
            # Generic pass-through; user is responsible for choosing a lossless
            # or visually lossless codec supported by both ffmpeg and imageProcessor.
            self.logger.info("Using custom intermediate codec: %s", codec)
            return ["-c:v", codec]

    def split_sbs_video(
        self,
        input_path: str,
        output_left: str,
        output_right: str,
        video_info: Dict,
    ) -> Tuple[bool, bool]:
        """Split SBS video into left/right eye streams using the intermediate codec."""
        self.logger.info("Splitting SBS video into left/right eye streams...")

        width = video_info["width"]
        height = video_info["height"]
        half_width = width // 2

        codec_args = self._get_intermediate_codec_args()

        # Left eye
        self.logger.info("Extracting left eye to %s", output_left)
        cmd_left = [
            "ffmpeg", "-y",
            "-v", "quiet",
            "-stats",
            "-i", input_path,
            "-vf", f"crop={half_width}:{height}:0:0",
            *codec_args,
            "-an",
            output_left,
        ]
        result_left = subprocess.run(cmd_left, stdout=None, stderr=None, text=True)
        if result_left.returncode != 0:
            self.logger.error("Failed to extract left eye")
            return False, False

        # Right eye
        self.logger.info("Extracting right eye to %s", output_right)
        cmd_right = [
            "ffmpeg", "-y",
            "-v", "quiet",
            "-stats",
            "-i", input_path,
            "-vf", f"crop={half_width}:{height}:{half_width}:0",
            *codec_args,
            "-an",
            output_right,
        ]
        result_right = subprocess.run(cmd_right, stdout=None, stderr=None, text=True)
        if result_right.returncode != 0:
            self.logger.error("Failed to extract right eye")
            return False, False

        self.logger.info("[OK] Successfully split SBS video")
        return True, True

    def extract_audio(self, input_path: str, output_path: str) -> bool:
        """
        Extract audio stream using codec copy.

        Audio is stored in a Matroska container (.mka) to support essentially
        any codec (AC-3, E-AC-3, DTS, TrueHD, etc.) without re-encoding.
        """
        self.logger.info("Extracting audio to %s", output_path)

        cmd = [
            "ffmpeg", "-y",
            "-v", "quiet",
            "-stats",
            "-i", input_path,
            "-vn", "-sn", "-dn",
            "-c:a", "copy",
            "-f", "matroska",
            output_path,
        ]

        result = subprocess.run(cmd, stdout=None, stderr=None, text=True)
        if result.returncode != 0:
            self.logger.warning("Failed to extract audio (may not exist)")
            return False

        self.logger.info("[OK] Successfully extracted audio")
        return True

    # -------------------------------------------------------------------------
    # Eye processing (imageProcessor)
    # -------------------------------------------------------------------------
    def _process_single_eye(
        self,
        input_file: str,
        output_file: str,
        video_info: Dict,
        eye: str,
        device: Optional[str] = None,
    ) -> bool:
        """Process a single eye through imageProcessor or copy in test mode."""
        test_mode = self.config.get("processing", {}).get("test_mode", False)

        if test_mode:
            self.logger.info("TEST MODE: Copying %s eye file (no imageProcessor processing)", eye)
            shutil.copy(input_file, output_file)
            return True

        self.logger.info("Starting imageProcessor processing for %s eye...", eye)
        cmd = self._build_imageProcessor_command(input_file, output_file, video_info, device)

        try:
            result = subprocess.run(cmd, stdout=None, stderr=None, text=True)
        except FileNotFoundError:
            self.logger.error("imageProcessor-cli executable not found when running for %s eye", eye)
            return False
        except Exception as e:  # pragma: no cover - generic safety net
            self.logger.error("Error executing imageProcessor for %s eye: %s", eye, e)
            return False

        if result.returncode != 0:
            self.logger.error("imageProcessor failed for %s eye with exit code %s", eye, result.returncode)
            return False

        self.logger.info("[OK] Successfully processed %s eye", eye)
        return True

    def process_both_eyes(
        self,
        left_file: str,
        right_file: str,
        left_output: str,
        right_output: str,
        video_info: Dict,
    ) -> Tuple[bool, bool]:
        """
        Process both eyes, optionally in parallel.

        Parallel mode avoids Python's multiprocessing overhead and instead
        launches two imageProcessor processes with subprocess.Popen.
        """
        parallel = self.config.get("processing", {}).get("parallel_eyes", True)
        test_mode = self.config.get("processing", {}).get("test_mode", False)

        # In test mode, we just do simple sequential copies
        if test_mode or not parallel:
            return self.process_both_eyes_sequential(left_file, right_file, left_output, right_output, video_info)

        self.logger.info("Processing left and right eyes in parallel...")

        imageProcessor_config = self.config["imageProcessor"]
        left_device = imageProcessor_config.get("left_eye_device") or imageProcessor_config.get("device", "cuda:0")
        right_device = imageProcessor_config.get("right_eye_device") or imageProcessor_config.get("device", "cuda:0")

        if left_device != right_device:
            self.logger.info("Using separate GPUs: Left=%s, Right=%s", left_device, right_device)

        # Build commands (for logging + Popen)
        left_cmd = self._build_imageProcessor_command(left_file, left_output, video_info, left_device)
        right_cmd = self._build_imageProcessor_command(right_file, right_output, video_info, right_device)

        def cmd_to_str(cmd: List[str]) -> str:
            return " ".join(f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd)

        if not test_mode:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("imageProcessor Command for Left Eye:")
            self.logger.info(cmd_to_str(left_cmd))
            self.logger.info("=" * 80)
            self.logger.info("imageProcessor Command for Right Eye:")
            self.logger.info(cmd_to_str(right_cmd))
            self.logger.info("=" * 80)
            self.logger.info("")

        # Launch both processes
        try:
            left_proc = subprocess.Popen(left_cmd)
            right_proc = subprocess.Popen(right_cmd)

            left_ret = left_proc.wait()
            right_ret = right_proc.wait()
        except Exception as e:  # pragma: no cover - generic safety net
            self.logger.error("Parallel imageProcessor execution failed: %s", e)
            self.logger.warning("Retrying with sequential processing...")
            return self.process_both_eyes_sequential(left_file, right_file, left_output, right_output, video_info)

        left_success = (left_ret == 0)
        right_success = (right_ret == 0)

        if not (left_success and right_success):
            self.logger.warning("Parallel processing had failures, retrying sequentially...")
            return self.process_both_eyes_sequential(left_file, right_file, left_output, right_output, video_info)

        return left_success, right_success

    def process_both_eyes_sequential(
        self,
        left_file: str,
        right_file: str,
        left_output: str,
        right_output: str,
        video_info: Dict,
    ) -> Tuple[bool, bool]:
        """Process both eyes sequentially."""
        self.logger.info("Processing eyes sequentially...")

        device = self.config["imageProcessor"].get("device", "cuda:0")

        left_success = self._process_single_eye(left_file, left_output, video_info, "Left", device)
        if not left_success:
            return False, False

        right_success = self._process_single_eye(right_file, right_output, video_info, "Right", device)
        return left_success, right_success

    # -------------------------------------------------------------------------
    # Temp file management & high-level operations
    # -------------------------------------------------------------------------
    def cleanup_temp_files(self, temp_dir: Path) -> None:
        """Clean up temporary files (unless keep_intermediates is enabled)."""
        keep_intermediates = self.config.get("processing", {}).get("keep_intermediates", False)
        if keep_intermediates:
            self.logger.info("Keeping intermediate files in %s", temp_dir)
            return

        self.logger.info("Cleaning up temporary files in %s", temp_dir)
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:  # pragma: no cover - best-effort cleanup
            self.logger.warning("Failed to clean up temp files: %s", e)

    def process_file(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Process a single VR SBS video file."""
        input_p = Path(input_path).resolve()
        if not input_p.exists():
            self.logger.error("Input file not found: %s", input_p)
            return False

        # Determine output path
        if output_path:
            output_p = Path(output_path).resolve()
        else:
            suffix = self.config.get("processing", {}).get("output_suffix", "-Restored")
            output_p = input_p.parent / f"{input_p.stem}{suffix}{input_p.suffix}"

        self.logger.info("Processing: %s", input_p.name)
        self.logger.info("Output: %s", output_p)

        # Temp directory handling
        processing_cfg = self.config.get("processing", {})
        temp_base = processing_cfg.get("temp_dir")

        if temp_base:
            # User-specified temp; recommend pointing this to a fast, large SSD/NVMe.
            base_dir = Path(temp_base)
        else:
            # System temp: e.g. C:\Users\<user>\AppData\Local\Temp\vrisber
            base_dir = Path(tempfile.gettempdir()) / "vrisber"

        base_dir.mkdir(parents=True, exist_ok=True)
        file_temp_dir = base_dir / input_p.stem
        file_temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Using temp directory: %s", file_temp_dir)
        self.logger.warning(
            "Ensure there is sufficient free space on the temp volume; "
            "lossless intermediates can require ~3–4× the input file size."
        )

        try:
            # Step 0: Analyze video
            video_info = self.analyze_video(str(input_p))

            # Temp file paths
            left_split = file_temp_dir / f"{input_p.stem}-L-Split.mkv"
            right_split = file_temp_dir / f"{input_p.stem}-R-Split.mkv"
            audio_file = file_temp_dir / f"{input_p.stem}-A.mka"
            left_restored = file_temp_dir / f"{input_p.stem}-L-Restored.mkv"
            right_restored = file_temp_dir / f"{input_p.stem}-R-Restored.mkv"

            # Step 1: Split SBS
            left_ok, right_ok = self.split_sbs_video(
                str(input_p),
                str(left_split),
                str(right_split),
                video_info,
            )
            if not (left_ok and right_ok):
                self.logger.error("Failed to split video into left/right eyes")
                return False

            # Step 2: Extract audio (best-effort)
            self.extract_audio(str(input_p), str(audio_file))

            # Step 3: Process both eyes through imageProcessor (or copy in test mode)
            left_ok, right_ok = self.process_both_eyes(
                str(left_split),
                str(right_split),
                str(left_restored),
                str(right_restored),
                video_info,
            )
            if not (left_ok and right_ok):
                self.logger.error("Failed to process eyes through imageProcessor")
                return False

            # Step 4: Final SBS mux
            success = self.mux_sbs_video(
                str(left_restored),
                str(right_restored),
                str(audio_file) if audio_file.exists() else None,
                str(output_p),
                video_info,
            )
            if not success:
                self.logger.error("Failed to mux final SBS video")
                return False

            self.logger.info("[DONE] Successfully processed %s", input_p.name)
            return True

        finally:
            self.cleanup_temp_files(file_temp_dir)

    def process_folder(self, folder_path: str, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """Process all supported files in a folder."""
        folder_p = Path(folder_path).resolve()
        if not folder_p.is_dir():
            self.logger.error("Not a directory: %s", folder_p)
            return {"success": [], "failed": [], "skipped": []}

        # Extensions (case-insensitive)
        extensions = self.config.get("batch", {}).get(
            "video_extensions",
            [".mp4", ".mkv", ".avi", ".mov", ".ts"],
        )
        extensions = [ext.lower() for ext in extensions]

        video_files: List[Path] = []
        for entry in folder_p.iterdir():
            if entry.is_file() and entry.suffix.lower() in extensions:
                video_files.append(entry)

        if not video_files:
            self.logger.warning("No video files found in %s", folder_p)
            return {"success": [], "failed": [], "skipped": []}

        self.logger.info("Found %d video file(s) to process", len(video_files))

        # Output directory
        if output_dir:
            out_dir_p = Path(output_dir).resolve()
            out_dir_p.mkdir(parents=True, exist_ok=True)
        else:
            out_dir_p = None

        results: Dict[str, List[str]] = {"success": [], "failed": [], "skipped": []}
        skip_existing = self.config.get("batch", {}).get("skip_existing", True)

        for idx, video_file in enumerate(sorted(video_files), 1):
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("Processing file %d/%d: %s", idx, len(video_files), video_file.name)
            self.logger.info("=" * 60)

            if out_dir_p:
                suffix = self.config.get("processing", {}).get("output_suffix", "-Restored")
                output_path = out_dir_p / f"{video_file.stem}{suffix}{video_file.suffix}"
            else:
                output_path = None

            # Skip if output exists
            if output_path and output_path.exists() and skip_existing:
                self.logger.info("Skipping (already exists): %s", output_path.name)
                results["skipped"].append(str(video_file))
                continue

            ok = self.process_file(str(video_file), str(output_path) if output_path else None)
            if ok:
                results["success"].append(str(video_file))
            else:
                results["failed"].append(str(video_file))

        self.logger.info("")
        self.logger.info("Batch processing complete:")
        self.logger.info("  Total:    %d", len(video_files))
        self.logger.info("  Success:  %d", len(results["success"]))
        self.logger.info("  Failed:   %d", len(results["failed"]))
        self.logger.info("  Skipped:  %d", len(results["skipped"]))

        return results

    # -------------------------------------------------------------------------
    # Final SBS mux
    # -------------------------------------------------------------------------
    def mux_sbs_video(
        self,
        left_file: str,
        right_file: str,
        audio_file: str,
        output_file: str,
        video_info: Dict,
    ) -> bool:
        """Mux left and right eye streams back into SBS format with audio."""
        self.logger.info("Muxing left/right eyes back into SBS format...")

        from pathlib import Path

        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]

        processing_cfg = self.config.get("processing", {})
        encoding_cfg = self.config.get("encoding", {})

        test_mode = bool(processing_cfg.get("test_mode", False))
        codec = encoding_cfg.get("codec", "hevc_nvenc")
        keyint_seconds = float(encoding_cfg.get("keyframe_interval_seconds", 5))
        keyint = max(1, int(round(fps * keyint_seconds)))

        # Base command and video inputs
        cmd: List[str] = [
            "ffmpeg",
            "-y",
            "-v",
            "info",   # let ffmpeg actually tell us what went wrong
            "-stats",
            "-i",
            left_file,
            "-i",
            right_file,
        ]

        has_audio = bool(audio_file and os.path.exists(audio_file))
        if has_audio:
            # Audio is the THIRD input (index 2)
            cmd.extend(["-i", audio_file])

        # Build the hstack filter AFTER all -i inputs
        # Both eye streams are square (e.g. 2160x2160), stacked horizontally
        filter_complex = "[0:v][1:v]hstack=inputs=2[v]"

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[v]"])

        if has_audio:
            # Audio comes from input #2 (the .mka file)
            cmd.extend(["-map", "2:a"])

        # ---- Video encoder settings (single lossy encode) ----
        # If you already have _build_encoder_options, use that; otherwise, inline.
        def _build_encoder_options(
            codec_name: str,
            cfg: Dict,
            keyint: int,
            fps_val: float,
        ) -> List[str]:
            fps_rounded = int(round(fps_val))

            if codec_name == "hevc_nvenc":
                preset = cfg.get("preset", "p7")
                rc = cfg.get("rc", "constqp")
                qp = cfg.get("qp", 18)
                rc_lookahead = cfg.get("rc_lookahead", 32)
                spatial_aq = cfg.get("spatial_aq", 1)
                temporal_aq = cfg.get("temporal_aq", 1)
                aq_strength = cfg.get("aq_strength", 8)
                bf = cfg.get("bf", 3)
                b_ref_mode = cfg.get("b_ref_mode", "middle")

                return [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    preset,
                    "-rc",
                    rc,
                    "-qp",
                    str(qp),
                    "-rc-lookahead",
                    str(rc_lookahead),
                    "-spatial_aq",
                    str(spatial_aq),
                    "-temporal_aq",
                    str(temporal_aq),
                    "-aq-strength",
                    str(aq_strength),
                    "-bf",
                    str(bf),
                    "-b_ref_mode",
                    b_ref_mode,
                    "-g",
                    str(keyint),
                    "-r",
                    str(fps_rounded),
                    "-pix_fmt",
                    "yuv420p",
                ]

            if codec_name == "hevc":
                crf = cfg.get("crf", 15)
                preset = cfg.get("preset", "slower")
                x265_params = cfg.get("x265_params", "")

                args = [
                    "-c:v",
                    "libx265",
                    "-crf",
                    str(crf),
                    "-preset",
                    preset,
                    "-x265-params",
                    f"keyint={keyint}:min-keyint={keyint}",
                ]
                if x265_params:
                    args[-1] += f":{x265_params}"
                return args

            if codec_name == "libsvtav1":
                crf = cfg.get("crf", 20)
                preset = cfg.get("preset", 4)
                return [
                    "-c:v",
                    "libsvtav1",
                    "-crf",
                    str(crf),
                    "-preset",
                    str(preset),
                    "-g",
                    str(keyint),
                    "-r",
                    str(int(round(fps_val))),
                ]

            # Fallback: H.264
            crf = cfg.get("crf", 18)
            preset = cfg.get("preset", "slow")
            return [
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-g",
                str(keyint),
                "-r",
                str(int(round(fps_val))),
            ]

        codec_cfg = encoding_cfg.get(codec, encoding_cfg.get(codec.lower(), {}))
        cmd.extend(_build_encoder_options(codec, codec_cfg, keyint, fps))

        # Audio copy (if present)
        if has_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])

        # Only apply faststart for MP4/MOV containers
        if encoding_cfg.get("moov_front", True):
            ext = Path(output_file).suffix.lower()
            if ext in {".mp4", ".m4v", ".mov"}:
                cmd.extend(["-movflags", "+faststart"])

        cmd.append(output_file)

        cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        self.logger.info("Running final mux/encode...")
        self.logger.debug(f"Mux command: {cmd_str}")

        result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            self.logger.error(
                f"Failed to mux video (exit code {result.returncode})"
            )
            if result.stderr:
                self.logger.error("ffmpeg stderr:")
                self.logger.error(result.stderr.strip())
            return False

        self.logger.info(f"[OK] Successfully created output: {output_file}")
        return True
