"""CosTaL.read_fcs: FCS files of every data layout the reader supports, against known values and,
where it is installed, against fcsparser (whose behavior it follows)."""

import warnings

import numpy as np
import pytest

import CosTaL


def write_fcs(path, data, *, datatype="F", byteord="1,2,3,4", widths=None, ranges=None, names_s=None,
              names_n=None, extra=None, delimiter="/"):
    """A minimal FCS 3.1 writer for the tests."""
    events, pars = data.shape
    endian = "<" if byteord in ("1,2,3,4", "1,2") else ">"
    widths = widths or [{"F": 32, "D": 64, "I": 16}[datatype]] * pars
    if datatype == "I":
        cols = []
        for j, w in enumerate(widths):
            b = w // 8
            v = data[:, j].astype(np.uint64)
            raw = np.stack([((v >> np.uint64(8 * i)) & np.uint64(255)).astype(np.uint8) for i in range(b)], axis=1)
            cols.append(raw if endian == "<" else raw[:, ::-1])
        body = np.concatenate(cols, axis=1).tobytes()
    else:
        body = data.astype(f"{endian}{'f4' if datatype == 'F' else 'f8'}").tobytes()
    names_n = names_n or [f"Ch{j + 1}" for j in range(pars)]
    names_s = names_s or [f"Marker {j + 1}" for j in range(pars)]
    ranges = ranges or [2 ** w for w in widths]
    kw = {"$BYTEORD": byteord, "$DATATYPE": datatype, "$MODE": "L", "$PAR": str(pars), "$TOT": str(events),
          "$NEXTDATA": "0"}
    for j in range(pars):
        kw.update({f"$P{j + 1}N": names_n[j], f"$P{j + 1}B": str(widths[j]), f"$P{j + 1}R": str(ranges[j]),
                   f"$P{j + 1}E": "0,0"})
        if names_s[j]:  # FCS has no empty values (two delimiters in a row are an escaped one): omit the key
            kw[f"$P{j + 1}S"] = names_s[j]
    kw.update(extra or {})
    esc = lambda s: s.replace(delimiter, delimiter * 2)  # noqa: E731
    text_start = 58
    for _ in range(3):  # the offsets change the TEXT length; iterate to a fixed point
        text = delimiter + "".join(f"{esc(k)}{delimiter}{esc(v)}{delimiter}" for k, v in kw.items())
        data_start = text_start + len(text.encode())
        kw["$BEGINDATA"], kw["$ENDDATA"] = str(data_start), str(data_start + len(body) - 1)
    text = delimiter + "".join(f"{esc(k)}{delimiter}{esc(v)}{delimiter}" for k, v in kw.items())
    data_start = text_start + len(text.encode())
    header = b"FCS3.1    " + b"".join(f"{v:>8}".encode() for v in (
        text_start, data_start - 1, data_start, data_start + len(body) - 1, 0, 0))
    path.write_bytes(header + text.encode() + body)
    return path


def check(path, expected, names):
    meta, df = CosTaL.read_fcs(path)
    assert list(df.columns) == names
    np.testing.assert_array_equal(df.to_numpy(), expected.astype(np.float32))
    try:
        import fcsparser
    except ImportError:
        return meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2, d2 = fcsparser.parse(str(path))
    assert list(d2.columns) == list(df.columns)
    np.testing.assert_array_equal(d2.to_numpy(), df.to_numpy())
    assert all(m2[k] == meta[k] for k in m2 if k != "__header__")
    return meta


@pytest.mark.parametrize("datatype", ["F", "D"])
@pytest.mark.parametrize("byteord", ["1,2,3,4", "4,3,2,1"])
def test_float_data(tmp_path, datatype, byteord):
    x = np.random.default_rng(0).normal(100, 50, (500, 4))
    meta = check(write_fcs(tmp_path / "f.fcs", x, datatype=datatype, byteord=byteord), x,
                 [f"Marker {j + 1}" for j in range(4)])
    assert meta["$PAR"] == 4 and meta["$TOT"] == 500


@pytest.mark.parametrize("byteord", ["1,2,3,4", "4,3,2,1"])
def test_integer_data_with_range_masks_and_mixed_widths(tmp_path, byteord):
    rng = np.random.default_rng(1)
    widths = [16, 32, 24, 16]
    ranges = [1024, 2 ** 32, 2 ** 24, 4096]
    raw = np.column_stack([rng.integers(0, 2 ** w, 300) for w in widths]).astype(np.uint64)
    expected = raw & np.array([2 ** int(np.ceil(np.log2(r))) - 1 for r in ranges], dtype=np.uint64)
    path = write_fcs(tmp_path / "i.fcs", raw, datatype="I", byteord=byteord, widths=widths, ranges=ranges)
    meta, df = CosTaL.read_fcs(path)
    np.testing.assert_array_equal(df.to_numpy(), expected.astype(np.float32))


def test_uniform_integer_data_matches_fcsparser(tmp_path):
    raw = np.random.default_rng(2).integers(0, 2 ** 16, (400, 5)).astype(np.uint64)
    ranges = [1024, 65536, 4096, 65536, 262144]
    expected = raw & np.array([2 ** int(np.ceil(np.log2(r))) - 1 for r in ranges], dtype=np.uint64)
    check(write_fcs(tmp_path / "u.fcs", raw, datatype="I", ranges=ranges), expected,
          [f"Marker {j + 1}" for j in range(5)])


def test_channel_naming_and_escaped_delimiters(tmp_path):
    x = np.arange(12, dtype=float).reshape(4, 3)
    # $PnS repeats → fall back to $PnN; a value containing the delimiter is escaped by doubling it
    path = write_fcs(tmp_path / "n.fcs", x, names_s=["CD4", "CD4", ""], names_n=["FL1-A", "FL2-A", "FL3-A"],
                     extra={"$FIL": "run/1.fcs"})
    with pytest.warns(UserWarning, match="not unique"):
        meta, df = CosTaL.read_fcs(path)
    assert list(df.columns) == ["FL1-A", "FL2-A", "FL3-A"] and meta["$FIL"] == "run/1.fcs"
    _, df = CosTaL.read_fcs(path, channel_naming="$PnN")
    assert list(df.columns) == ["FL1-A", "FL2-A", "FL3-A"]
    path = write_fcs(tmp_path / "m.fcs", x, names_s=["CD4", "", "CD8"], names_n=["FL1-A", "FL2-A", "FL3-A"])
    assert list(CosTaL.read_fcs(path)[1].columns) == ["CD4", "FL2-A", "CD8"]  # empty $PnS → $PnN


def test_rejects_what_it_cannot_read(tmp_path):
    bad = tmp_path / "bad.fcs"
    bad.write_bytes(b"not an fcs file at all" * 10)
    with pytest.raises(ValueError, match="not an FCS file"):
        CosTaL.read_fcs(bad)
    x = np.ones((10, 2))
    path = write_fcs(tmp_path / "t.fcs", x)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        CosTaL.read_fcs(path)
