"""CosTaL+ 命令行入口（骨架）。

交付形态之一是"不写 Python 也能用"（docs/01 §3）。现在只有 `info`，
用来在没有 Python REPL 的环境里确认 wheel 装对了。
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
    sub.add_parser("info", help="打印构建信息，确认扩展模块装对了")

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
