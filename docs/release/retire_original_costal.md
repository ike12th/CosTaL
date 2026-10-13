# Retiring the original CosTaL

The original package lives at https://github.com/li000678/CosTaL and on PyPI as `CosTaL` (last
release 0.0.2.4). CosTaL+ replaces it. Retiring it takes three steps, all by the maintainer, after
CosTaL+ is on PyPI.

## 1. Point the old repository here

Replace the README of `li000678/CosTaL` with the text below (adjust the repository URL if CosTaL+
moves), commit, then *Settings → Archive this repository*. Archiving keeps every link and the history
readable while making clear that the code is no longer maintained.

```markdown
# CosTaL has moved to CosTaL+

**This repository is archived.** CosTaL is maintained as **CosTaL+**:
https://github.com/ike12th/costal-plus

    pip install "costal-plus[all]"

CosTaL+ is the same method (Li et al., *Briefings in Bioinformatics* 2023,
doi:10.1093/bib/bbad157) — exact cosine kNN, Tanimoto edge weights, Leiden — rewritten in Rust:

- same clusters as the published method, tens to hundreds of times faster;
- every platform (macOS, Linux, Windows), no external binary, no temporary files;
- a command line: `costal-plus cluster cells.fcs --platform mass`;
- opt-in extras for rare populations (`find_communities_multiscale`).

Existing scripts keep working: `pip install "CosTaL>=0.1"` installs a thin compatibility package that
runs `CosTaL.find_communities`, `CosTaL.pp`, `CosTaL.clustering` and `CosTaL.return_scores` on
CosTaL+, or switch the import to `from costal_plus import legacy as CosTaL`. See the
[migration guide](https://github.com/ike12th/costal-plus/blob/main/docs/user_guide.md#11-migrating-from-costal-002).

Please cite the paper as before.
```

## 2. Publish the compatibility package (optional, recommended)

`packaging/costal-compat/` is a pure-Python `CosTaL` 0.1.0 that depends on `costal-plus` and
re-exports `costal_plus.legacy`, with a `FutureWarning` pointing to CosTaL+. Publishing it under the
old name means `pip install --upgrade CosTaL` moves users to CosTaL+ without breaking their scripts:

```bash
cd packaging/costal-compat
python -m build            # pip install build
twine upload dist/*        # needs the maintainer's PyPI credentials for the CosTaL project
```

## 3. Optionally yank nothing

Leave 0.0.2.4 on PyPI (pinned environments and the paper's reproduction depend on it); the
compatibility release supersedes it for everyone who upgrades.
