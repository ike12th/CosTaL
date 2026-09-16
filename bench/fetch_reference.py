#!/usr/bin/env python3
"""把 profile 需要的两样东西拉齐：原版 CosTaL 源码 + Levine_32dim 数据集。

一键复现（docs/01 §4）的第一环：跑 profile 之前先跑这个。
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REF = HERE / "reference"

COSTAL_VERSION = "0.0.2.4"
LEVINE = {
    "Levine_32dim.fcs": "https://raw.githubusercontent.com/lmweber/"
    "benchmark-data-Levine-32-dim/master/data/Levine_32dim.fcs",
    "Levine_32dim_notransform.fcs": "https://raw.githubusercontent.com/lmweber/"
    "benchmark-data-Levine-32-dim/master/data/Levine_32dim_notransform.fcs",
}


def fetch_costal() -> Path:
    """下原版 CosTaL 的 sdist 并解包。不 pip install：它的 install_requires 里
    有已经从 PyPI 下架的 `sklearn` 占位包，装不上；profile 只需要能 import。"""
    target = REF / f"CosTaL-{COSTAL_VERSION}"
    if target.is_dir():
        return target
    REF.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", f"CosTaL=={COSTAL_VERSION}",
         "--no-deps", "--no-binary", ":all:", "-d", str(REF)],
        check=True,
    )
    sdist = next(REF.glob(f"CosTaL-{COSTAL_VERSION}.tar.gz"))
    with tarfile.open(sdist) as tf:
        tf.extractall(REF)
    return target


def fetch_levine() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in LEVINE.items():
        dst = DATA / name
        if dst.exists():
            continue
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlopen  # noqa: B018 - 明确走 urllib，无额外依赖
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            f.write(r.read())


if __name__ == "__main__":
    print("CosTaL source:", fetch_costal())
    fetch_levine()
    print("data dir     :", DATA)
