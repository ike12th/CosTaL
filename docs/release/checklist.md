# Release checklist

Releases are cut from `main`. Publishing to PyPI and making the repository public are outward-facing
and irreversible: they are done by the maintainer, never by an automated job on its own.

## 1. Prepare (in the repository)

- [ ] Version in `Cargo.toml` (`[workspace.package] version` and the `costal-core` dependency) and in
      `pyproject.toml` — all three equal.
- [ ] `CHANGELOG.md`: the release's section with its date; anything that changes a default is a
      major version (docs/04 §6.5).
- [ ] `CITATION.cff`: `version`.
- [ ] `docs/user_guide.md` matches the API (`python -c "import costal_plus, inspect; ..."` spot checks;
      the examples in `examples/` run in the test suite).
- [ ] Local: `scripts/dev.sh` passes (fmt, clippy, `cargo test`, pytest).

## 2. Verify on every platform

- [ ] Actions → `CI` → *Run workflow* with `all_os` ticked: Linux, macOS, Windows green.
- [ ] Actions → `wheels` → *Run workflow*: all 8 wheels and the sdist build, and every native wheel
      installs and passes its tests on its runner. Download the artifact `wheels-*` to inspect.

## 3. Tag

```bash
git tag -a v1.0.0 -m "CosTaL+ 1.0.0"
git push origin v1.0.0
```

The tag re-runs `CI` (all platforms) and `wheels`.

## 4. Publish to PyPI (maintainer)

One-time setup: on pypi.org create the project `costal-plus` through a *pending trusted publisher*
(owner/repository of this GitHub repository, workflow `wheels.yml`, environment `pypi`), and in the
repository settings create the environment `pypi` (optionally with a required reviewer).

Then on GitHub: *Releases → Draft a new release* from the tag, paste the CHANGELOG section, *Publish*.
Publishing the release runs the `publish` job of `wheels.yml`, which uploads the wheels and the sdist
built for that tag. Check https://pypi.org/project/costal-plus/ and, in a fresh environment:

```bash
pip install "costal-plus[all]" && costal-plus info && python -c "import costal_plus; print(costal_plus.__version__)"
```

Without trusted publishing: download the `wheels-*` artifacts into `dist/` and run
`twine upload dist/*`.

## 5. Retire the original CosTaL

See [retire_original_costal.md](retire_original_costal.md).
