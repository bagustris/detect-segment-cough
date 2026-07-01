"""Configuration file support for coughkit.

Config files are TOML.  Search order (first found wins):
  1. $COUGHKIT_CONFIG environment variable
  2. ./coughkit.toml  (project-local override)
  3. ~/.config/coughkit/config.toml  (user-global)

CLI flags always override config values; config values override built-in defaults.
"""

import os
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

_DEFAULT_PATH = Path.home() / ".config" / "coughkit" / "config.toml"
_LOCAL_PATH   = Path("coughkit.toml")

TEMPLATE = """\
# coughkit configuration
# CLI flags always override these values.
# Uncomment and edit the lines you want to change.

[count]
# threshold    = 0.5      # probability threshold to classify a segment as a cough
# duration     = 30.0     # max mic recording duration in seconds (omit for unlimited)
# fs_out       = 16000    # sample rate for loading / recording
# verbose      = false    # print per-event probability while counting
# no_mic_level = false    # suppress the live RMS bar during mic recording

[segment]
# output_dir = "./segments"   # directory for segmented .wav files
# fs_out     = 16000
"""


def find_config_path():
    """Return the first config file that exists, or None."""
    env = os.environ.get("COUGHKIT_CONFIG")
    if env:
        return Path(env)
    if _LOCAL_PATH.exists():
        return _LOCAL_PATH
    if _DEFAULT_PATH.exists():
        return _DEFAULT_PATH
    return None


def load_config(path=None):
    """Load and return the config as a nested dict.

    *path* overrides auto-discovery.  Returns an empty dict when no file is
    found or when tomllib / tomli is unavailable.
    """
    if tomllib is None:
        return {}
    p = Path(path) if path else find_config_path()
    if p is None or not p.exists():
        return {}
    try:
        with p.open("rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        print(f"[coughkit] warning: could not read config {p}: {exc}",
              file=sys.stderr)
        return {}


def apply_config(args, section, cfg=None):
    """Fill *args* attributes that are still None from the config *section*.

    Attribute names are derived from config keys by replacing ``-`` with ``_``.
    """
    if cfg is None:
        cfg = load_config()
    for key, value in cfg.get(section, {}).items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)


def init_config(path=None):
    """Write the template config to *path* (default: ``~/.config/coughkit/config.toml``).

    Returns the path written to.
    """
    dest = Path(path) if path else _DEFAULT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(TEMPLATE)
    return dest
