#!/usr/bin/env python3
"""Fetch the two things the profile needs: the original CosTaL source + the Levine_32dim data set.

The first link of one-command reproduction (docs/01 §4): run this before profiling.
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
    """Download and unpack the original CosTaL sdist. We don't pip install it: its install_requires
    lists the `sklearn` placeholder package that has been removed from PyPI, so it can't be
    installed; the profile only needs to import it."""
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
        urllib.request.urlopen  # noqa: B018 - deliberately urllib, no extra dependency
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            f.write(r.read())


if __name__ == "__main__":
    print("CosTaL source:", fetch_costal())
    fetch_levine()
    print("data dir     :", DATA)
