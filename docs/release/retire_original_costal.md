# Pointing the 1.0 repository to CosTaL 2.0

CosTaL 1.0 lives at https://github.com/li000678/CosTaL. CosTaL 2.0 is the next version of the same
package: it is published to the same PyPI project (`CosTaL`), so `pip install --upgrade CosTaL`
moves users to 2.0, and 1.0 code keeps running on it. What remains is the old repository.

## Replace its README and archive it

This can be done before 2.0 reaches PyPI: the README below sends readers to this repository's own
installation instructions instead of naming a `pip` command that could go stale. As the owner of
`li000678/CosTaL`, on GitHub:

1. Edit `README.md` on the default branch (`master`), replace its content with the text below
   (adjust the URL if this repository moves), and commit.
2. *Settings → General*: set the description to "CosTaL 1.0 (archived). Continued at
   github.com/ike12th/CosTaL" and the website to https://github.com/ike12th/CosTaL.
3. *Settings → General → Danger Zone → Archive this repository*. Archiving keeps every link and the
   history readable, and marks the code as no longer maintained. (An archived repository is
   read-only; unarchive it first to change anything later.)

```markdown
# CosTaL 1.0 — retired

**This repository is no longer maintained.** It holds CosTaL 1.0 (on PyPI as 0.0.2.4).
CosTaL continues at **https://github.com/ike12th/CosTaL** — see there to install the current
version.

CosTaL 2.0 is the same method (Li et al., *Briefings in Bioinformatics* 2023,
doi:10.1093/bib/bbad157) — exact cosine kNN, Tanimoto edge weights, Leiden — rewritten in Rust:

- the same clusters as the published method, tens to hundreds of times faster;
- every platform (macOS, Linux, Windows), no external binary, no temporary files;
- 1.0 code runs unchanged (`import CosTaL`, `CosTaL.find_communities(X, pp_method="mass")`,
  `CosTaL.clustering`, `CosTaL.return_scores`, …); new code can use `import costal`;
- a command line: `costal cluster cells.fcs --platform mass`;
- opt-in extras for rare populations (`find_communities_multiscale`).

Please cite the paper as before.
```

0.0.2.4 stays on PyPI for anyone who pins it.
