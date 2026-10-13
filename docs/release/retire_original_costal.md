# Pointing the 1.0 repository to CosTaL 2.0

CosTaL 1.0 lives at https://github.com/li000678/CosTaL. CosTaL 2.0 is the next version of the same
package: it is published to the same PyPI project (`CosTaL`), so `pip install --upgrade CosTaL`
moves users to 2.0, and 1.0 code keeps running on it. What remains is the old repository.

## Replace its README and archive it

Once 2.0 is on PyPI, replace the README of `li000678/CosTaL` with the text below (adjust the URL if
this repository moves or is renamed), commit, then *Settings → Archive this repository*. Archiving
keeps every link and the history readable and marks the old code as no longer maintained.

```markdown
# CosTaL — now developed at https://github.com/ike12th/costal-plus

**This repository holds CosTaL 1.0 and is archived.** CosTaL 2.0 and later are developed at
https://github.com/ike12th/costal-plus.

    pip install --upgrade CosTaL

CosTaL 2.0 is the same method (Li et al., *Briefings in Bioinformatics* 2023,
doi:10.1093/bib/bbad157) — exact cosine kNN, Tanimoto edge weights, Leiden — rewritten in Rust:

- the same clusters as the published method, tens to hundreds of times faster;
- every platform (macOS, Linux, Windows), no external binary, no temporary files;
- 1.0 code runs unchanged (`CosTaL.find_communities(X, pp_method="mass")`, `CosTaL.clustering`,
  `CosTaL.return_scores`, …);
- a command line: `costal cluster cells.fcs --platform mass`;
- opt-in extras for rare populations (`find_communities_multiscale`).

Please cite the paper as before.
```

0.0.2.4 stays on PyPI for anyone who pins it.
