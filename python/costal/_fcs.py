"""``read_fcs()``: FCS 2.0 / 3.0 / 3.1 list-mode files, read the way fcsparser reads them.

CosTaL used fcsparser, but its current release pins numpy below 2, which makes every environment that
installs it fall back to old numpy (and fail where no such wheel exists). This reader follows
fcsparser 0.2's behavior field for field — header and $BEGINDATA offsets, the TEXT segment with escaped
delimiters, $DATATYPE F / D / I in either byte order, $PnR bit masks on integer data, mixed widths,
$NEXTDATA data sets, and the channel naming rule ($PnS, falling back to $PnN when $PnS is empty or not
unique) — and returns the same ``(meta, DataFrame)``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_fcs"]

_ORDERS = {"1,2,3,4": "<", "1,2": "<", "4,3,2,1": ">", "2,1": ">"}


def _header(buf: bytes, offset: int) -> dict:
    head = {"FCS format": buf[offset:offset + 6]}
    fields = ("text start", "text end", "data start", "data end", "analysis start", "analysis end")
    for i, name in enumerate(fields):
        raw = buf[offset + 10 + 8 * i:offset + 18 + 8 * i]
        try:
            value = int(raw)
        except ValueError:
            value = 0
        head[name] = value + offset
    if head["text end"] == head["data start"]:  # seen in the wild; as fcsparser does
        head["text end"] -= 1
    if not head["FCS format"].startswith(b"FCS"):
        raise ValueError("not an FCS file (no FCS header)")
    for k in ("text start", "text end"):
        if head[k] == 0 or head[k] > len(buf):
            raise ValueError(f"the FCS file seems corrupted: cannot locate the {k!r} of its TEXT segment")
    return head


def _text(raw: str) -> dict:
    """The TEXT segment as a dict; a doubled delimiter is an escaped one inside a key or value."""
    delimiter = raw[0]
    if raw[-1] != delimiter:
        if delimiter.strip() == delimiter:
            raw = raw.strip()
        if raw[-1] != delimiter:
            warnings.warn("the TEXT segment does not end with its delimiter; the file may be read incorrectly",
                          stacklevel=3)
            raw = raw[1:]
        else:
            raw = raw[1:-1]
    else:
        raw = raw[1:-1]
    parts = [chunk.split(delimiter) for chunk in raw.split(delimiter * 2)]
    elements = parts[0]
    for chunk in parts[1:]:
        elements[-1] += delimiter + chunk[0]
        elements.extend(chunk[1:])
    return dict(zip(elements[0::2], elements[1::2]))


def _columns(raw: np.ndarray, widths: list[int], endian: str) -> list[np.ndarray]:
    """Unsigned integers of arbitrary byte widths from (events × record) bytes."""
    out, pos = [], 0
    for w in widths:
        block = raw[:, pos:pos + w].astype(np.uint64)
        order = range(w) if endian == "<" else range(w - 1, -1, -1)
        value = np.zeros(raw.shape[0], dtype=np.uint64)
        for shift, byte in enumerate(order):
            value |= block[:, byte] << np.uint64(8 * shift)
        out.append(value)
        pos += w
    return out


def read_fcs(path: str | Path, *, channel_naming: str = "$PnS", data_set: int = 0,
             dtype: str | None = "float32", encoding: str = "utf-8") -> tuple[dict, Any]:
    """Read an FCS file: ``(meta, data)``, as ``fcsparser.parse(path)`` returns them.

    ``meta`` holds the TEXT keywords (``$PAR``, ``$TOT``, ``$PnB`` and ``$NEXTDATA`` as integers) and
    ``__header__``; ``data`` is a pandas DataFrame, events × channels, named by ``$PnS`` (or ``$PnN``
    where ``$PnS`` is empty, or for all channels when the ``$PnS`` names repeat), cast to ``dtype``.
    """
    import pandas as pd

    if channel_naming not in ("$PnS", "$PnN"):
        raise ValueError('channel_naming must be "$PnS" or "$PnN"')
    buf = Path(path).read_bytes()

    offset, seen = 0, 0
    while True:
        head = _header(buf, offset)
        raw_text = buf[head["text start"]:head["text end"] + 1]
        try:
            text = _text(raw_text.decode(encoding))
        except UnicodeDecodeError:
            text = _text(raw_text.decode(encoding, errors="ignore"))
        nxt = int(text.get("$NEXTDATA", "0") or 0)
        if seen == data_set or nxt == 0:
            if seen != data_set:
                warnings.warn("the file holds fewer data sets than requested; reading the last one", stacklevel=2)
            break
        seen += 1
        offset += nxt

    pars = int(text["$PAR"])
    events = int(text["$TOT"])
    if "$P0B" in text:
        raise ValueError("FCS parameters numbered from 0 are not supported")
    channels = range(1, pars + 1)
    for key in [f"$P{c}B" for c in channels] + ["$PAR", "$TOT"]:
        text[key] = int(text[key])
    if "$NEXTDATA" in text:
        text["$NEXTDATA"] = int(text["$NEXTDATA"])

    names_n = [text.get(f"$P{c}N", "") for c in channels]
    names_s = [text.get(f"$P{c}S", "") or text.get(f"$P{c}N", "") for c in channels]
    names = names_s if channel_naming == "$PnS" else names_n
    if len(set(names)) != len(names):
        warnings.warn(f"the {channel_naming} channel names are not unique; using the alternate names",
                      stacklevel=2)
        names = names_n if channel_naming == "$PnS" else names_s

    meta = dict(text)
    meta["__header__"] = head
    if text.get("$MODE", "L") != "L":
        raise ValueError(f"FCS mode {text.get('$MODE')!r} is not supported (list mode only)")
    order = text.get("$BYTEORD", "").strip()
    if order not in _ORDERS:
        raise ValueError(f"FCS byte order {order!r} is not supported")
    endian = _ORDERS[order]
    kind = text.get("$DATATYPE", "")
    if kind not in ("F", "D", "I"):
        raise ValueError(f"FCS data type {kind!r} is not supported")

    start = head["data start"]
    if start == offset:  # header field 0: the DATA segment starts past 99,999,999 bytes, see $BEGINDATA
        start = int(text["$BEGINDATA"]) + offset
    widths = [text[f"$P{c}B"] // 8 for c in channels]
    record = sum(widths)
    if start + events * record > len(buf):
        raise ValueError("the FCS file is truncated: its DATA segment is incomplete")
    if kind in ("F", "D"):
        width = {"F": 4, "D": 8}[kind]
        if any(w != width for w in widths):
            raise ValueError(f"$DATATYPE {kind} with $PnB other than {8 * width} is not supported")
        values = np.frombuffer(buf, dtype=f"{endian}f{width}", count=events * pars, offset=start).reshape(events, pars)
    else:
        block = np.frombuffer(buf, dtype=np.uint8, count=events * record, offset=start).reshape(events, record)
        cols = _columns(block, widths, endian)
        for i, c in enumerate(channels):
            bits = math.ceil(math.log2(float(text[f"$P{c}R"])))
            if bits < 64:
                cols[i] &= np.uint64(2 ** bits - 1)
        values = np.column_stack(cols) if cols else np.zeros((events, 0), dtype=np.uint64)
    # one copy: native byte order and the requested dtype (the values fcsparser gives)
    values = values.astype(np.dtype(dtype) if dtype else values.dtype.newbyteorder("="))
    return meta, pd.DataFrame(values, columns=names, copy=False)
