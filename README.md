# Vrisber

**Vrisber** is a VR Side-by-Side (SBS) video processor that sits on top of an external image restoration CLI (for example, `lada-cli`).  

It takes an SBS VR video (e.g. 3840×2160), splits it into left/right eye streams, runs each eye through a configurable restoration backend, and reconstructs a high-quality SBS output tuned for VR headsets like the Quest 2/3.

Design philosophy:

> **Quality first.** Lossless intermediates, one lossy encode, and knobs where they matter.

Vrisber is implemented in pure Python (standard library only) and delegates heavy work to:

- `ffmpeg` – split, audio extraction, final encode/mux  
- `ffprobe` – video analysis  
- `imageProcessor-cli` – per-eye restoration (mosaic removal, etc.)

---

## Features

- 🎥 **VR-aware SBS pipeline**
  - Splits 3D SBS into per-eye videos.
  - Reassembles a proper SBS output with the correct dimensions and aspect.

- 🧠 **Pluggable restoration backend**
  - Config block `imageProcessor` controls the external tool: path, device, models, extra args.
  - Optional dual-GPU setup (left eye on GPU0, right eye on GPU1).
  - Parallel or sequential eye processing.

- 🛡️ **Quality-first pipeline**
  - Lossless (or effectively lossless) intermediates using a configurable codec (default: FFV1).
  - Exactly **one lossy encode**: the final SBS mux step.

- ⚙️ **Flexible final encoding**
  - HEVC NVENC (`hevc_nvenc`)
  - Software HEVC (`hevc` / x265)
  - AV1 (`libsvtav1`)
  - H.264 (`h264` / libx264)  
  All controlled via the `encoding` block in `config.json`.

- 🧵 **Parallel & batch processing**
  - Optional parallel eye processing (two backend processes in flight).
  - Folder mode with “skip existing output” so you can safely re-run batches.

- 📜 **Logging & test mode**
  - Logs to `vrisber.log`.
  - `--test-mode` validates split/mux without running the restoration backend.

---

## Disk & temp space

Vrisber uses **lossless intermediates** for:

- Split eye streams (`*-L-Split.mkv`, `*-R-Split.mkv`)
- Restored eye streams (`*-L-Restored.mkv`, `*-R-Restored.mkv`)

This is intentional: all heavy processing stages are lossless, and only the final encode is lossy.

Trade-off:

> Expect peak temp usage of roughly **3–4× the input file size**.

Configuration:

- Set `processing.temp_dir` in `config.json` to a fast, spacious SSD / NVMe volume.  
- Vrisber creates a subdirectory per input file under that base, e.g. `D:\VrisberTemp\MyVideo`.

---

## Requirements

- **OS**: Windows or Linux (developed and tested primarily on Windows).
- **Python**: 3.8+.
- **External tools**:
  - `ffmpeg` (with FFV1 support).
  - `ffprobe` (bundled with ffmpeg).
  - An image restoration CLI, referenced as `imageProcessor-cli`:
    - In practice this is typically configured to point at `lada-cli.exe` or similar.
- **Hardware**:
  - Fast SSD/NVMe with multiple times the input size free (for intermediates).
  - NVIDIA GPU recommended for NVENC (`hevc_nvenc`).
  - 8 GB+ RAM.

The Python package itself uses **only the standard library**; no third-party dependencies.

---

## Installation

Clone the repo and install in editable mode:

~~~bash
git clone https://github.com/<your-username>/vrisber.git
cd vrisber

# Optional but recommended: virtualenv
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

python -m pip install -e .
~~~

This installs a `vrisber` console command, wired to `vrisber.cli:main`.

---

## Configuration

All knobs live in `config.json`. An example is provided as `config.example.json`.

Typical first-time setup:

~~~bash
# Linux/macOS:
cp config.example.json config.json

# Windows:
copy config.example.json config.json
~~~

Then edit `config.json` and set at least:

- `imageProcessor.path` → path to your CLI executable (e.g. `lada-cli.exe`).  
- `imageProcessor.device` → default device, for example `"cuda:0"`.  
- `encoding.codec` → final output codec (`"hevc_nvenc"`, `"hevc"`, `"libsvtav1"`, `"h264"`).  
- `processing.temp_dir` → a fast, large temp directory (e.g. `D:\VrisberTemp`).

### Config overview

High-level structure (simplified):

~~~jsonc
{
  "imageProcessor": {
    "path": "imageProcessor-cli.exe",
    "device": "cuda:0",
    "left_eye_device": null,
    "right_eye_device": null,
    "max_clip_length": 120,
    "mosaic_restoration_model": null,
    "mosaic_detection_model_path": null,
    "extra_args": ""
  },

  "encoding": {
    "codec": "hevc_nvenc",
    "keyframe_interval_seconds": 5,
    "moov_front": true,
    "hevc_nvenc": { },
    "hevc":       { },
    "libsvtav1":  { },
    "h264":       { }
  },

  "processing": {
    "parallel_eyes": true,
    "test_mode": false,
    "intermediate_codec": "ffv1",
    "temp_dir": null,
    "keep_intermediates": false,
    "output_suffix": "-Restored"
  },

  "batch": {
    "video_extensions": [".mp4", ".mkv", ".avi", ".mov", ".ts"],
    "skip_existing": true
  },

  "logging": {
    "level": "INFO"
  }
}
~~~

Notes:

- `encoding.*` controls **only the final SBS encode**.  
- `processing.intermediate_codec` controls both the split eye streams and backend outputs (lossless intermediates).
- `processing.temp_dir` should point to a large, fast volume; if `null`, system temp is used (e.g. `C:\Users\<user>\AppData\Local\Temp\vrisber`).

---

## Quickstart

### 1. Sanity check external tools

From a terminal:

~~~bash
ffmpeg -version
ffprobe -version
imageProcessor-cli --help   # or lada-cli --help if that’s what you’re using
~~~

Make sure all three are installed and on your PATH.

### 2. Smoke test (no restoration)

Validate the split/mux pipeline first:

~~~bash
vrisber path/to/your_sbs_video.mp4 --test-mode
~~~

- Splits into left/right eyes.
- Copies each eye (no processing).
- Muxes back into SBS output with original audio.

The output should be visually identical to the input. If this fails, it’s usually:

- `ffmpeg`/`ffprobe` not found,
- bad `processing.temp_dir`,
- permission issues on the temp disk.

### 3. Full run with restoration

Once test-mode works, run the full pipeline:

~~~bash
# Use default config.json, output next to input
vrisber path/to/your_sbs_video.mp4

# Explicit output path
vrisber path/to/your_sbs_video.mp4 path/to/output_restored.mp4
~~~

Vrisber will:

1. Analyze video with `ffprobe`.  
2. Split SBS into `*-L-Split.mkv` / `*-R-Split.mkv` (lossless).  
3. Extract audio to `*-A.mka` (codec copy).  
4. Run `imageProcessor-cli` per eye.  
5. Rebuild SBS and encode with the final codec.  

Watch the console / `vrisber.log` – you’ll see the exact commands used for each eye and for the final mux.

---

## Usage

### Single file

~~~bash
# Default behavior (output next to input, suffix from config)
vrisber input.mp4

# Explicit output path
vrisber input.mp4 output.mp4
~~~

### Batch mode (folders)

~~~bash
# Process all supported files in ./videos
vrisber ./videos/

# Process into a separate output directory
vrisber ./videos/ --output-dir ./restored/
~~~

Batch mode respects:

- `batch.video_extensions` for which files to pick up.
- `batch.skip_existing` to avoid re-processing files where the output already exists.

### Command-line overrides

The CLI exposes a few useful overrides:

~~~bash
# Override device just for this run
vrisber input.mp4 --device cuda:1

# Override final codec
vrisber input.mp4 --codec hevc        # software HEVC
vrisber input.mp4 --codec libsvtav1   # AV1 via libsvtav1

# Disable/enable parallel eye processing regardless of config.json
vrisber input.mp4 --no-parallel
vrisber input.mp4 --parallel

# Use an alternate config file
vrisber input.mp4 --config another_config.json
~~~

You can also combine these with folder mode:

~~~bash
vrisber ./videos/ --output-dir ./restored/ --codec hevc_nvenc --device cuda:0
~~~

---

## How the pipeline works (Option A)

For a 3D SBS input like `VR_SBS.mp4` (e.g. 3840×2160 @ 60 fps), the pipeline is:

1. **Analyze** – `analyze_video` calls `ffprobe` to get width/height/fps/codec/pix_fmt/bitrate.  
2. **Split eyes** – `split_sbs_video` crops the left and right halves into two lossless `.mkv` intermediates using `processing.intermediate_codec` (FFV1 by default).  
3. **Extract audio** – `extract_audio` copies the audio stream into an `.mka` container without re-encoding.  
4. **Per-eye restoration** – `process_both_eyes` runs the configured backend (`imageProcessor-cli`) on each eye, optionally in parallel and/or on separate GPUs.  
5. **Final SBS mux** – `mux_sbs_video` stacks the two restored eye streams with `hstack`, maps video + audio, and performs a single final encode using `encoding.codec` and its sub-config.  
6. **Cleanup** – Temp directory is deleted unless `processing.keep_intermediates` is `true`.

If the intermediate codec is lossless and supported by both ffmpeg and your backend, then:

- All heavy stages (split, restoration) are lossless.
- Only the final encode introduces generational loss.

---

## Logging & troubleshooting

- Logs go to `vrisber.log` in the working directory.  
- Log level is controlled by `logging.level` in `config.json` (e.g. `INFO`, `DEBUG`).

Common issues:

- **`ffmpeg not found` / `ffprobe not found`**  
  → Install ffmpeg and make sure it’s on your PATH.

- **`imageProcessor-cli not found`**  
  → Check `imageProcessor.path`, or verify it’s on PATH by running `imageProcessor-cli --help`.

- **Out of GPU memory**
  - Disable parallel eye processing via `processing.parallel_eyes = false` or `--no-parallel`.  
  - Reduce `imageProcessor.max_clip_length` so the backend works on shorter segments.

- **Disk space issues**
  - Point `processing.temp_dir` at a bigger/faster disk.  
  - Keep `keep_intermediates = false` (the default).

---

## Versioning

Current state: **baseline release (0.5.0)** – this repository snapshot is the starting point for future entries.  
Future releases will be documented in `CHANGELOG.md`.

---

## License

Vrisber is released under the [Apache License 2.0](LICENSE).
