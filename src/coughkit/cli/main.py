#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coughkit command-line interface."""

import argparse

from coughkit.cli import count as count_cli
from coughkit.cli import detect as detect_cli
from coughkit.cli import segment as segment_cli
from coughkit.cli.common import add_version_argument
from coughkit.config import (apply_config, init_config, find_config_path,
                             TEMPLATE, _DEFAULT_PATH)


def _run_detect(args):
    detect_cli.detect(args.file)


def _run_segment(args):
    apply_config(args, 'segment')
    args.output_dir = args.output_dir or './'
    args.fs_out     = args.fs_out     or 16000
    segment_cli.segment(args.file, args.output_dir, args.fs_out)


def _run_count(args):
    apply_config(args, 'count')
    args.fs_out       = args.fs_out       if args.fs_out       is not None else 16000
    args.threshold    = args.threshold    if args.threshold    is not None else count_cli.DEFAULT_THRESHOLD
    args.verbose      = args.verbose      or False
    args.no_mic_level = args.no_mic_level or False
    count_cli.count(input_file=args.file, use_mic=args.mic,
                    duration=args.duration, fs_out=args.fs_out,
                    threshold=args.threshold, verbose=args.verbose,
                    show_mic_level=not args.no_mic_level)


def _run_config(args):
    if args.config_cmd == 'init':
        dest = init_config(args.path)
        print(f"Config template written to: {dest}")
    elif args.config_cmd == 'show':
        p = find_config_path()
        if p is None:
            print("No config file found.  Run 'coughkit config init' to create one.")
            print(f"Default location: {_DEFAULT_PATH}")
        else:
            print(f"Config file: {p}")
            print(p.read_text())
    else:
        print("Usage: coughkit config <init|show>")


# (name, module-or-None, runner, help) for each subcommand.
_SUBCOMMANDS = [
    ("detect",  detect_cli,  _run_detect,
     "Detect whether an audio file contains a cough."),
    ("segment", segment_cli, _run_segment,
     "Segment a recording into individual cough WAV files."),
    ("count",   count_cli,   _run_count,
     "Count cough events in an audio file or microphone recording."),
]


def build_parser(prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_version_argument(parser)

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND",
                                       required=True)
    for name, module, runner, help_text in _SUBCOMMANDS:
        sub = subparsers.add_parser(name, help=help_text,
                                    description=module.__doc__)
        add_version_argument(sub)
        module.add_arguments(sub)
        sub.set_defaults(func=runner)

    # config subcommand
    cfg_parser = subparsers.add_parser(
        "config", help="Manage the coughkit config file.")
    cfg_subs = cfg_parser.add_subparsers(dest="config_cmd", metavar="ACTION")
    init_p = cfg_subs.add_parser("init", help="Write a template config file.")
    init_p.add_argument("--path", default=None,
                        help="Destination path (default: ~/.config/coughkit/config.toml)")
    cfg_subs.add_parser("show", help="Print the active config file and its path.")
    cfg_parser.set_defaults(func=_run_config)

    return parser


def main(argv=None, prog=None):
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
