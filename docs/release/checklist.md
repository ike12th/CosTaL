# Release checklist

Releases are cut from `main`. Publishing to PyPI and making the repository public are outward-facing
and irreversible: they are done by the maintainer, never by an automated job on its own.

## 1. Prepare (in the repository)

- [ ] Version in `Cargo.toml` (`[workspace.package] version` and the `costal-core` dependency) and in
      `pyproject.toml` — all three equal.
- [ ] `CHANGELOG.md`: the release's section with its date; anything that changes a default is a
      major version (docs/04 §6.5).
- [ ] `CITATION.cff`: `version`.
- [ ] `docs/user_guide.md` matches the API (`python -c "import CosTaL, inspect; ..."` spot checks;
      the examples in `examples/` run in the test suite).
- [ ] Local: `scripts/dev.sh` passes (fmt, clippy, `cargo test`, pytest).

## 2. Verify on every platform

- [ ] Actions → `CI` → *Run workflow* with `all_os` ticked: Linux, macOS, Windows green.
- [ ] Actions → `wheels` → *Run workflow*: all 8 wheels and the sdist build, and every native wheel
      installs and passes its tests on its runner. Download the artifact `wheels-*` to inspect.
      (Run 38471935477, before the renumbering to 2.0.0: all green; 63 passed on macOS arm64,
      Linux x86-64 and Windows x64; 52 passed / 11 skipped on Windows arm64, where leidenalg and
      scanpy have no wheels.)

## 3. Tag

```bash
git tag -a v2.0.0 -m "CosTaL 2.0.0"
git push origin v2.0.0
```

The tag re-runs `CI` (all platforms) and `wheels`.

## 4. Publish to PyPI (maintainer)

One-time setup: the PyPI project is the existing `CosTaL` (0.0.2.4 and earlier). As its owner, on
pypi.org → *Your projects → CosTaL → Settings → Publishing*, add a trusted publisher: this GitHub
repository (owner and name), workflow `wheels.yml`, environment `pypi`. In the repository settings,
create the environment `pypi` (optionally with a required reviewer).

Then on GitHub: *Releases → Draft a new release* from the tag, paste the CHANGELOG section, *Publish*.
Publishing the release runs the `publish` job of `wheels.yml`, which uploads the wheels and the sdist
built for that tag. Check https://pypi.org/project/CosTaL/ and, in a fresh environment:

```bash
pip install --upgrade CosTaL && costal info && python -c "import CosTaL; print(CosTaL.__version__)"
```

Without trusted publishing: download the `wheels-*` artifacts into `dist/` and run
`twine upload dist/*`.

## 5. Point the 1.0 repository here

See [retire_original_costal.md](retire_original_costal.md).
