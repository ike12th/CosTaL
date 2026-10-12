"""CosTaL+ command-line entry point (skeleton).

One delivery form is "usable without writing Python" (docs/01 §3). For now there is only
`info`, to confirm the wheel installed correctly where no Python REPL is at hand.
"""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__, core_version, hello


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="costal-plus", description="CosTaL+ CLI")
    parser.add_argument("--version", action="version", version=f"costal-plus {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print build info to confirm the extension module is installed correctly")

    args = parser.parse_args(argv)
    if args.cmd in (None, "info"):
        print(hello())
        print(f"wheel version : {__version__}")
        print(f"core version  : {core_version()}")
        print(f"python        : {sys.version.split()[0]}")
        print(f"platform      : {platform.platform()} / {platform.machine()}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
