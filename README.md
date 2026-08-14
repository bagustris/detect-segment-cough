

# Detect and Segment Cough
This repository hosts code and models to **detect**, **segment**, and **count** cough sounds. Detecting a cough returns the probability that an audio file contains cough sounds. Segmenting coughs writes separate WAV files for individual detected cough events. Counting classifies each segment and reports the total number of coughs — including in real time from a microphone.


# Input-output

- Input: audio files (`.wav`) that may contain one or more cough sounds, or live microphone input.
- Output:
  - Detection: cough probability for the input file.
  - Segmentation: new WAV files containing individual cough segments.
  - Count: number of cough events in the input file or live microphone recording.

# **Supported Python Version and Model** (IMPORTANT!)

The current `pyproject.toml` targets modern Python versions (>=3.11) and uses CPU-only XGBoost via `xgboost-cpu`.

The original bundled classifier pickle (`models/cough_classifier`) was produced with XGBoost 0.90, which cannot be unpickled by modern XGBoost. Runtime detection now loads `models/cough_classifier_migrated.json`, a stable model-IO export migrated from that legacy pickle. The scaler (`models/cough_classification_scaler`) is still an older `scikit-learn` pickle, so Python may warn when loading it with newer packages. Treat model outputs from a newer runtime as compatibility-sensitive and validate them against known recordings before production use.

# Installation

Install the package (and its dependencies) into a virtual environment with pip or uv. An editable install (`-e`) is recommended so the bundled models in `models/` resolve correctly:

```
# Inference-only (minimal dependencies):
pip install -e .
# or
uv pip install -e .

# With development tools (pytest, tqdm):
pip install -e '.[dev]'

# With training tools (pandas, required for scripts/train_classifier.py):
pip install -e '.[train]'
```

This installs the `coughkit` package plus the primary `coughkit` console command. The older `cough-detect`, `cough-segment`, and `cough-count` commands are still installed as compatibility aliases. The core dependencies use `xgboost-cpu`, so GPU-only packages such as `nvidia-nccl-cu12` are not required for CPU inference.

## Command Line Usage

```bash
# Show the installed toolkit version:
coughkit --version

# Detect cough (probability that the file contains a cough):
coughkit detect -f input_file.wav

# Segment coughs into a chosen output directory:
coughkit segment -f input_file.wav -o output_segments/

# Count cough events in a file:
coughkit count -f input_file.wav

# Count cough events from the microphone (real-time, stop with Ctrl+C):
coughkit count -m

# Count with a time limit and verbose per-event output:
coughkit count -m -d 60 -v
```

The compatibility aliases remain available, e.g. `cough-detect -f input_file.wav`. The top-level command is also runnable as a module, e.g. `python -m coughkit detect -f input_file.wav`.

### Microphone mode

When using `-m`, a live RMS energy bar is shown so you can verify your microphone is picking up signal before coughing:

```
Recording… (press Ctrl+C to stop)
  mic: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0241  |  2 cough(s)
```

The count updates in real time after each detected cough event. Pass `--no-mic-level` to suppress the bar.

Detection uses a streaming energy VAD: audio is accumulated into events when signal power exceeds 2× the estimated background level, and classified when 300 ms of silence follows. This means each cough is counted immediately after it ends, without waiting for Ctrl+C.

## Configuration file

Persistent defaults can be stored in a TOML config file so you do not have to repeat flags on every run. CLI flags always take precedence over config values.

```bash
# Create the default config at ~/.config/coughkit/config.toml
coughkit config init

# Or create a project-local override (takes precedence over the global one)
coughkit config init --path ./coughkit.toml

# Show which config file is active and its contents
coughkit config show
```

Example `coughkit.toml`:

```toml
[count]
threshold    = 0.7      # raise to reduce false positives (default: 0.5)
duration     = 60.0     # stop mic recording after 60 s
fs_out       = 16000
verbose      = true

[segment]
output_dir = "./segments"
fs_out     = 16000
```

The config file is searched in this order:
1. `$COUGHKIT_CONFIG` environment variable
2. `./coughkit.toml` (project-local)
3. `~/.config/coughkit/config.toml` (user-global)

## Python Usage
```python
from coughkit import classify_cough, load_cough_classifier, load_scaler
from scipy.io import wavfile

input_file = './data/sample_recordings/cough.wav'

model = load_cough_classifier()
scaler = load_scaler()

fs, x = wavfile.read(input_file)
prob = classify_cough(x, fs, model, scaler)
print(f"{input_file} has probability of cough: {prob}")
```

# Example
```
# Detect cough probability:
coughkit detect -f data/sample_recordings/cough.wav
data/sample_recordings/cough.wav has probability of cough: 0.995194137096405

# Segment into individual cough files:
coughkit segment -f data/sample_recordings/cough.wav -o output_segments/
Write to output_segments/cough-0.wav

# Count coughs in a file:
coughkit count -f data/sample_recordings/cough.wav
data/sample_recordings/cough.wav: 1 cough(s) detected

# Count from microphone (real-time):
coughkit count -m
Recording… (press Ctrl+C to stop)
  mic: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0121  |  1 cough(s)
^C
microphone: 1 cough(s) detected
```

You may see an `InconsistentVersionWarning` from `scikit-learn` when loading the legacy scaler. This warning is expected with modern environments and does not prevent detection from running.

# Overview

In the wake of the COVID-19 pandemic, mass coronavirus testing has proven essential to governments in monitoring the spread of the disease, isolating infected individuals, and effectively "flattening the curve" of infections over time. However, this oropharyngeal swab test is physically invasive and must be performed by a trained clinician. This requires patients to travel to a laboratory facility to get tested, thereby potentially infecting others along the way. Ideally, testing would be performed noninvasively at no cost, and administered at the homes of potential patients to minimize contamination risk.

The World Health Organization (WHO) has reported that 67.7% of COVID-19 patients exhibit a "dry cough," which may be audibly different from coughs caused by other pathologies. Such cough sounds analysis has proven successful in diagnosing respiratory conditions like pertussis, asthma, and pneumonia.

At the Embedded Systems Laboratory (ESL) at EPFL, we have developed the COUGHVID database, which is an extensive dataset of COVID-19 cough sounds from around the world, partially validated by expert pulmonologists. We contribute our data, signal preprocessing source code, cough classification algorithm, and feature extraction methods to assist the global research community in developing algorithms to automatically screen for COVID-19 based on cough sounds.

# Data access

The COUGHVID dataset can be downloaded from the following Zenodo link: https://doi.org/10.5281/zenodo.4048311

## Notebooks
The `coughvid_classification_example.ipynb` notebook illustrates the usage of the cough classifier model for removing unwanted recordings from a cough database.

The `segmentation_and_SNR_example.ipynb` notebook is an example of how to use the automatic cough segmentation and SNR estimation algorithm.

## Source code

The reusable library lives in `src/coughkit/`:

- `coughkit/dsp.py` — signal preprocessing plus `classify_cough` (filter → 68-feature extraction → scale → XGBoost probability).
- `coughkit/features.py` — the `features` class computing all audio features used for cough classification.
- `coughkit/segmentation.py` — `segment_cough` (hysteresis comparator) and `compute_SNR`.
- `coughkit/models.py` — loaders for the bundled classifier and scaler.
- `coughkit/audio_io.py` — COUGHVID `.webm`/`.ogg` → `.wav` conversion (requires ffmpeg), microphone capture, and live RMS level rendering.
- `coughkit/config.py` — TOML config file loading and `apply_config` helper.
- `coughkit/cli/` — the `coughkit` command and legacy `cough-detect`, `cough-segment`, and `cough-count` entry points.

Offline tooling lives in `scripts/`:

- `scripts/train_classifier.py` — reproduce the classifier + scaler from the COUGHVID dataset.
- `scripts/convert_model_to_json.py` — migrate the legacy XGBoost 0.90 pickle to modern JSON.

## Models

- `models/cough_classifier_migrated.json`: modern XGBoost model loaded at runtime.
- `models/cough_classifier`: original legacy XGBoost 0.90 pickle, kept for provenance/backward compatibility.
- `models/cough_classification_scaler`: feature scaler used before classification.

Use `coughkit.load_cough_classifier()` instead of loading `models/cough_classifier` directly. Loading the legacy pickle with modern XGBoost will fail.


# Citation

When using this resource, please cite the following publication:

1. Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Sci Data* **8,** 156 (2021). https://doi.org/10.1038/s41597-021-00937-4

2. B. T. Atmaja, Zanjabila, Suyanto, and A. Sasou, "Comparing hysteresis comparator and RMS threshold methods for automatic single cough segmentations," Int. J. Inf. Technol., no. 0123456789, Dec. 2023, doi: 10.1007/s41870-023-01626-8.

3. https://c4science.ch/diffusion/10770/  (original repo forked from)
