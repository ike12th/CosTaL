"""CosTaL+ — CosTaL 的高性能重写。

方法来源：Li Y, Nguyen J, Anastasiu DC, Arriaga EA. *CosTaL: an accurate and
scalable graph-based clustering algorithm for high-dimensional single-cell data
analysis.* Briefings in Bioinformatics, 2023. doi:10.1093/bib/bbad157

当前是骨架阶段：只导出自检函数，算法尚未实现。
"""

from ._costal_plus import __version__, core_version, hello, row_l2_norms

__all__ = ["__version__", "core_version", "hello", "row_l2_norms"]
