# Changelog

## 0.3.0 - 01/07/2026  
- Added real-time mic counting with streaming VAD, live RMS energy bar, TOML config file support
- Changed -i/--input to -f/--file flag for all subcommands.

## 0.2.0 - 03/06/2026

- Modernize project organization into an installable `src`-layout `coughkit` package with reusable audio I/O, DSP, feature, segmentation, and model modules.
- Add the unified `coughkit` command with `detect`, `segment`, and `count` subcommands while keeping the legacy `cough-detect`, `cough-segment`, and `cough-count` entry points.
- Update runtime packaging for modern Python environments, including CPU-only XGBoost and the migrated JSON classifier model.
- Add CLI coverage for version flags and subcommand dispatch.

## 21/03/2022

- Submit paper to interspeech about cough segmentation (rejected), commit: f330c2fb90431c736ee495b668ac0b0e0994b0cf.

## 08/02/2022

- Rename repo from `detect-cough` to `detect-segment-cough`.

## 03/02/2022

- Initial version, forked from EPFL's original repo.
