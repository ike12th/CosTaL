"""A minimal reader for R's XDR serialization (.rda / .rds), for data only.

R's own ``load()`` can run code from a crafted file (promises are forced on access; CVE-2024-27322,
fixed in R 4.4.0), and the pure-Python ``rdata`` package can't read the namespace references inside
Bioconductor S4 objects. This parser follows R's ``serialize.c`` (``ReadItem``, ``ReadBC``) and only
rebuilds the object tree: nothing is evaluated, closures and byte code are kept as inert nodes.

    obj = read_rda(path)["Samusik_01_SE"]      # an RObject tree
    obj.attr("assays")                          # S4 slots are attributes

Vectors become numpy arrays (logical/integer: int32 with R's NA as -2**31; double: float64), strings
become lists of ``str``/``None``. Pairlists and attribute lists become ``dict``s keyed by tag.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NA_INTEGER = -(2 ** 31)

# SEXP types (Rinternals.h) and serialization pseudo-types (serialize.c)
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP, SPECIALSXP, BUILTINSXP, CHARSXP = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP, DOTSXP, VECSXP, EXPRSXP = 10, 13, 14, 15, 16, 17, 19, 20
BCODESXP, EXTPTRSXP, WEAKREFSXP, RAWSXP, S4SXP = 21, 22, 23, 24, 25
ALTREP_SXP, ATTRLISTSXP, ATTRLANGSXP = 238, 239, 240
BASEENV_SXP, EMPTYENV_SXP, BCREPREF, BCREPDEF, GENERICREFSXP, CLASSREFSXP = 241, 242, 243, 244, 245, 246
PERSISTSXP, PACKAGESXP, NAMESPACESXP, BASENAMESPACE_SXP = 247, 248, 249, 250
MISSINGARG_SXP, UNBOUNDVALUE_SXP, GLOBALENV_SXP, NILVALUE_SXP, REFSXP = 251, 252, 253, 254, 255
PAIRLIKE = (LISTSXP, LANGSXP, CLOSXP, PROMSXP, DOTSXP)


@dataclass
class RObject:
    type: int
    value: Any = None
    attributes: dict = field(default_factory=dict)
    tag: Any = None

    def attr(self, name: str) -> Any:
        return self.attributes.get(name)


@dataclass(frozen=True)
class Symbol:
    name: str


class _Reader:
    def __init__(self, data: bytes):
        self.buf = memoryview(data)
        self.pos = 0
        self.refs: list = []

    def int(self) -> int:
        (v,) = struct.unpack_from(">i", self.buf, self.pos)
        self.pos += 4
        return v

    def take(self, n: int) -> bytes:
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return bytes(out)

    def length(self) -> int:
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def array(self, n: int, fmt: str, width: int) -> np.ndarray:
        out = np.frombuffer(self.buf, dtype=fmt, count=n, offset=self.pos).astype(fmt[1:])
        self.pos += n * width
        return out

    def string_vec(self) -> list:
        if self.int() != 0:
            raise ValueError("bad string vector")
        return [self.item() for _ in range(self.int())]

    def attributes(self, obj: Any) -> dict:
        """An attribute pairlist as {tag: value}."""
        out = {}
        node = obj
        while isinstance(node, RObject) and node.type == LISTSXP:
            out[node.tag.name if isinstance(node.tag, Symbol) else str(node.tag)] = node.value[0]
            node = node.value[1]
        return out

    def item(self) -> Any:  # noqa: C901 - mirrors ReadItem's switch
        flags = self.int()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == NILVALUE_SXP:
            return None
        if t in (EMPTYENV_SXP, BASEENV_SXP, GLOBALENV_SXP, UNBOUNDVALUE_SXP, MISSINGARG_SXP, BASENAMESPACE_SXP):
            return RObject(t)
        if t == REFSXP:
            idx = flags >> 8 or self.int()
            return self.refs[idx - 1]
        if t in (PERSISTSXP, PACKAGESXP, NAMESPACESXP):
            obj = RObject(t, self.string_vec())
            self.refs.append(obj)
            return obj
        if t == SYMSXP:
            name = self.item()
            sym = Symbol(name.value if isinstance(name, RObject) else str(name))
            self.refs.append(sym)
            return sym
        if t == ENVSXP:
            locked = self.int()
            env = RObject(ENVSXP, {"locked": locked})
            self.refs.append(env)
            env.value.update(enclos=self.item(), frame=self.item(), hashtab=self.item())
            env.attributes = self.attributes(self.item())
            return env
        if t in PAIRLIKE:
            # iterate along the cdr chain so long pairlists don't recurse
            head = node = None
            while True:
                attrs = self.attributes(self.item()) if has_attr else {}
                tag = self.item() if has_tag else None
                car = self.item()
                cur = RObject(t, [car, None], attrs, tag)
                if node is None:
                    head = cur
                else:
                    node.value[1] = cur
                node = cur
                flags = self.int()
                nt = flags & 0xFF
                if nt not in PAIRLIKE:
                    self.pos -= 4
                    node.value[1] = self.item()
                    return head
                t, has_attr, has_tag = nt, bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == ALTREP_SXP:
            info, state, attr = self.item(), self.item(), self.item()
            return RObject(ALTREP_SXP, {"info": info, "state": state}, self.attributes(attr))
        if t == EXTPTRSXP:
            obj = RObject(t)
            self.refs.append(obj)
            obj.value = {"prot": self.item(), "tag": self.item()}
        elif t == WEAKREFSXP:
            obj = RObject(t)
            self.refs.append(obj)
        elif t in (SPECIALSXP, BUILTINSXP):
            obj = RObject(t, self.take(self.int()).decode())
        elif t == CHARSXP:
            n = self.int()
            obj = RObject(t, None if n == -1 else self.take(n).decode("utf-8", "replace"))
        elif t in (LGLSXP, INTSXP):
            obj = RObject(t, self.array(self.length(), ">i4", 4))
        elif t == REALSXP:
            obj = RObject(t, self.array(self.length(), ">f8", 8))
        elif t == CPLXSXP:
            obj = RObject(t, self.array(2 * self.length(), ">f8", 8))
        elif t == STRSXP:
            obj = RObject(t, [self.item().value for _ in range(self.length())])
        elif t in (VECSXP, EXPRSXP):
            obj = RObject(t, [self.item() for _ in range(self.length())])
        elif t == BCODESXP:
            reps = [None] * self.int()
            obj = self.bc1(reps)
        elif t == RAWSXP:
            obj = RObject(t, self.take(self.length()))
        elif t == S4SXP:
            obj = RObject(t)
        else:
            raise ValueError(f"unsupported SEXP type {t} at byte {self.pos - 4}")
        if has_attr:
            attrs = self.attributes(self.item())
            if t != CHARSXP:
                obj.attributes = attrs
        return obj

    def bc1(self, reps: list) -> RObject:
        code = self.item()
        consts = []
        for _ in range(self.int()):
            ct = self.int()
            if ct == BCODESXP:
                consts.append(self.bc1(reps))
            elif ct in (LANGSXP, LISTSXP, BCREPDEF, BCREPREF, ATTRLANGSXP, ATTRLISTSXP):
                consts.append(self.bc_lang(ct, reps))
            else:
                consts.append(self.item())
        return RObject(BCODESXP, {"code": code, "consts": consts})

    def bc_lang(self, t: int, reps: list) -> Any:
        if t == BCREPREF:
            return reps[self.int()]
        if t not in (BCREPDEF, LANGSXP, LISTSXP, ATTRLANGSXP, ATTRLISTSXP):
            return self.item()
        pos = -1
        if t == BCREPDEF:
            pos, t = self.int(), self.int()
        has_attr = t in (ATTRLANGSXP, ATTRLISTSXP)
        t = {ATTRLANGSXP: LANGSXP, ATTRLISTSXP: LISTSXP}.get(t, t)
        obj = RObject(t, [None, None])
        if pos >= 0:
            reps[pos] = obj
        if has_attr:
            obj.attributes = self.attributes(self.item())
        obj.tag = self.item()
        obj.value[0] = self.bc_lang(self.int(), reps)
        obj.value[1] = self.bc_lang(self.int(), reps)
        return obj


def _decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


def _stream(data: bytes) -> _Reader:
    if data[:2] != b"X\n":
        raise ValueError("only the XDR (binary, big-endian) format is supported")
    r = _Reader(data)
    r.pos = 2
    version = r.int()
    r.int(), r.int()  # writer and minimal reader versions
    if version == 3:
        r.take(r.int())  # native encoding
    return r


def read_rda(path) -> dict:
    """The objects saved in an .rda / .RData file, as {name: RObject}."""
    data = _decompress(open(path, "rb").read())
    if data[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise ValueError("not an .rda file")
    r = _stream(data[5:])
    return r.attributes(r.item())


def read_rds(path) -> Any:
    return _stream(_decompress(open(path, "rb").read())).item()


def vector(obj: Any) -> Any:
    """A plain value: numpy array, list of str, or a factor's labels; ALTREP compact sequences expanded."""
    if obj is None:
        return None
    if obj.type == ALTREP_SXP:
        cls = obj.value["info"].value[0]
        state = obj.value["state"]
        if isinstance(cls, Symbol) and cls.name in ("compact_intseq", "compact_realseq"):
            n, start, step = state.value[:3]
            return start + step * np.arange(int(n))
        if isinstance(cls, Symbol) and cls.name.startswith("wrap_"):
            return vector(state.value[0])
        if isinstance(cls, Symbol) and cls.name == "deferred_string":
            return [str(v) for v in vector(state.value[0])]
        raise ValueError(f"unsupported ALTREP class {cls}")
    levels = obj.attr("levels")
    if levels is not None and obj.type == INTSXP:
        lv = levels.value
        return [None if i == NA_INTEGER else lv[i - 1] for i in obj.value]
    return obj.value


def env_vars(env: RObject) -> dict:
    """The variables of a serialized environment (its frame pairlist and/or hash table)."""
    out = {}
    chains = [env.value.get("frame")]
    hashtab = env.value.get("hashtab")
    if isinstance(hashtab, RObject):
        chains += hashtab.value
    for node in chains:
        while isinstance(node, RObject) and node.type == LISTSXP:
            out[node.tag.name] = node.value[0]
            node = node.value[1]
    return out


def summarized_experiment(se: RObject) -> tuple[np.ndarray, dict, dict]:
    """The first assay matrix (rows × columns), rowData and colData of a Bioconductor
    SummarizedExperiment, as numpy / lists."""
    assays = env_vars(se.attr("assays").attr(".xData"))[".->data"].attr("listData")
    m = assays.value[0]
    nr, nc = vector(m.attr("dim"))
    x = np.ascontiguousarray(m.value.reshape(nc, nr).T)  # R stores column-major

    def frame(df: RObject) -> dict:
        ld = df.attr("listData")
        return {k: vector(v) for k, v in zip(vector(ld.attr("names")), ld.value)}

    return x, frame(se.attr("elementMetadata")), frame(se.attr("colData"))
