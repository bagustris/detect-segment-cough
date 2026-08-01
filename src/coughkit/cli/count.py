#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count cough events in an audio file or live microphone recording.

Pipeline: segment high-energy regions → classify each segment → count those
above a probability threshold.
"""

import sys
import warnings
from pathlib import Path

import librosa
import numpy as np

from audiokit import mic_stream, render_mic_level
from coughkit.cli import common
from coughkit.dsp import classify_cough
from coughkit.models import load_cough_classifier, load_scaler
from coughkit.segmentation import segment_cough

DEFAULT_THRESHOLD = 0.5

# preprocess_cough() always resamples to 2 * its 6 kHz cutoff internally;
# loading below that starves the classifier of content it was trained on.
MIN_RELIABLE_FS = 12000


def _warn_if_low_fs(fs_out):
    if fs_out < MIN_RELIABLE_FS:
        warnings.warn(
            f"fs_out={fs_out} is below {MIN_RELIABLE_FS} Hz, the sampling rate "
            "the cough classifier's feature pipeline expects; classification "
            "accuracy may degrade significantly.",
            stacklevel=3,
        )

# Streaming VAD parameters
_HIGH_MULT    = 2.0          # chunk power must exceed this × background to start event
_LOW_MULT     = 0.1          # chunk power must fall below this × background to be "silent"
_END_SILENCE  = 0.3          # seconds of silence required to close an event
_MIN_EVENT    = 0.2          # minimum event length in seconds
_MAX_EVENT    = 2.0          # safety cap: force-close an event this long even without silence
                              # (kept short - the classified buffer is real
                              # cough + trailing ambient audio, and dilution
                              # from trailing noise pulls its probability down)
_CALIBRATION  = 0.5          # seconds of leading audio used to seed background power
_EMA_ALPHA    = 0.05         # background power EMA update rate


def _load_file(input_file, fs_out=16000):
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    _warn_if_low_fs(fs_out)
    x, fs = librosa.load(str(input_path), sr=fs_out)
    return x, fs


def _count_mic_streaming(duration=None, fs_out=16000, threshold=DEFAULT_THRESHOLD,
                         verbose=False, show_mic_level=True):
    """Real-time mic counting using a per-chunk streaming VAD.

    Accumulates high-energy chunks into events.  When the signal drops below
    the low threshold for _END_SILENCE seconds, the event is classified.
    Background power is first seeded from _CALIBRATION seconds of leading
    audio (a fixed 1e-5 seed is far below real-room ambient power, ~1e-4,
    which would latch the detector open for the whole session), then tracked
    with an EMA so the thresholds keep adapting to ambient noise.  A
    _MAX_EVENT cap force-closes runaway events as a safety net.
    """
    _warn_if_low_fs(fs_out)
    model = load_cough_classifier()
    scaler = load_scaler()

    end_sil_samples   = int(_END_SILENCE * fs_out)
    min_event_samples = int(_MIN_EVENT * fs_out)
    max_event_samples = int(_MAX_EVENT * fs_out)
    calib_samples     = int(_CALIBRATION * fs_out)
    limit_samples     = int(duration * fs_out) if duration is not None else None

    bg_power      = None
    calib_powers  = []
    calib_count   = 0
    in_event      = False
    event_buf     = np.zeros(0, dtype=np.float32)
    silence_count = 0
    total         = 0
    recorded      = 0

    print("Recording… (press Ctrl+C to stop)" +
          (f" [max {duration}s]" if duration is not None else ""))

    def _maybe_count(audio):
        nonlocal total
        prob = classify_cough(audio, fs_out, model, scaler)
        if prob >= threshold:
            total += 1
            if verbose:
                sys.stdout.write(f"\n  cough detected (p={prob:.3f})\n")
                sys.stdout.flush()

    try:
        for chunk in mic_stream(capture_rate=fs_out, chunk_size=0.1):
            recorded += len(chunk)
            cp = float(np.mean(chunk ** 2))

            if show_mic_level:
                render_mic_level(chunk, suffix=f"{total} cough(s)")

            if bg_power is None:
                calib_powers.append(cp)
                calib_count += len(chunk)
                if calib_count < calib_samples:
                    if limit_samples is not None and recorded >= limit_samples:
                        break
                    continue
                # Calibration window just closed: seed bg_power and let this
                # same chunk fall through to normal event detection below.
                bg_power = max(float(np.mean(calib_powers)), 1e-6)

            if not in_event:
                if cp > _HIGH_MULT * bg_power:
                    in_event = True
                    event_buf = chunk.copy()
                    silence_count = 0
                else:
                    bg_power = (1 - _EMA_ALPHA) * bg_power + _EMA_ALPHA * cp
            else:
                event_buf = np.concatenate([event_buf, chunk])
                if len(event_buf) >= max_event_samples:
                    _maybe_count(event_buf)
                    in_event = False
                    event_buf = np.zeros(0, dtype=np.float32)
                    silence_count = 0
                    bg_power = (1 - _EMA_ALPHA) * bg_power + _EMA_ALPHA * cp
                elif cp < _LOW_MULT * bg_power:
                    silence_count += len(chunk)
                    if silence_count >= end_sil_samples:
                        if len(event_buf) >= min_event_samples:
                            _maybe_count(event_buf)
                        in_event = False
                        event_buf = np.zeros(0, dtype=np.float32)
                        silence_count = 0
                        bg_power = ((1 - _EMA_ALPHA) * bg_power
                                    + _EMA_ALPHA * cp)
                else:
                    silence_count = 0

            if limit_samples is not None and recorded >= limit_samples:
                break
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Flush any event still open when recording stopped
    if in_event and len(event_buf) >= min_event_samples:
        _maybe_count(event_buf)

    print(f"microphone: {total} cough(s) detected")
    return total


def count(input_file=None, use_mic=False, duration=None, fs_out=16000,
          threshold=DEFAULT_THRESHOLD, verbose=False, show_mic_level=True):
    """Return the number of segments classified as a cough above *threshold*."""
    if use_mic:
        return _count_mic_streaming(duration=duration, fs_out=fs_out,
                                    threshold=threshold, verbose=verbose,
                                    show_mic_level=show_mic_level)
    if input_file is None:
        raise ValueError("Provide input_file or set use_mic=True.")

    x, fs = _load_file(input_file, fs_out=fs_out)
    model = load_cough_classifier()
    scaler = load_scaler()
    segments, _ = segment_cough(x, fs, cough_padding=0.2)

    cough_count = 0
    for i, seg in enumerate(segments):
        prob = classify_cough(seg, fs, model, scaler)
        is_cough = prob >= threshold
        if is_cough:
            cough_count += 1
        if verbose:
            label = "cough" if is_cough else "not cough"
            print(f"  segment {i+1}: {len(seg)/fs:.2f}s  p={prob:.3f}  [{label}]")

    print(f"{input_file}: {cough_count} cough(s) detected")
    return cough_count


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('-f', '--file', metavar='FILE',
                        help='Path to input audio file')
    source.add_argument('-m', '--mic', action='store_true',
                        help='Record from the default microphone (stop with Ctrl+C)')

    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='Max recording duration in seconds for --mic '
                             '(default: unlimited, stop with Ctrl+C)')
    parser.add_argument('-fs', '--fs_out', type=int, default=None,
                        help='Sampling rate for loading/recording (default: 16000)')
    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help=f'Probability threshold for classifying a segment as a '
                             f'cough (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('-v', '--verbose', action='store_true', default=None,
                        help='Print per-segment probability and classification')
    parser.add_argument('--no-mic-level', action='store_true', default=None,
                        help='Suppress the live RMS energy bar during microphone capture')
    return parser


def build_parser(prog=None):
    return common.build_parser(add_arguments, __doc__, prog=prog)


def main(argv=None):
    from coughkit.config import apply_config
    args = build_parser().parse_args(argv)
    apply_config(args, 'count')
    # Apply built-in defaults for anything still None after config
    args.fs_out      = args.fs_out      if args.fs_out      is not None else 16000
    args.threshold   = args.threshold   if args.threshold   is not None else DEFAULT_THRESHOLD
    args.verbose     = args.verbose     or False
    args.no_mic_level = args.no_mic_level or False
    count(input_file=args.file, use_mic=args.mic, duration=args.duration,
          fs_out=args.fs_out, threshold=args.threshold, verbose=args.verbose,
          show_mic_level=not args.no_mic_level)


if __name__ == '__main__':
    main()
