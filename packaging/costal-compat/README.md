# CosTaL → CosTaL+

CosTaL is now maintained as **CosTaL+** (`pip install costal-plus`,
https://github.com/ike12th/costal-plus): the same method, rewritten in Rust — the same clusters,
tens to hundreds of times faster, on every platform, with no external binary.

This package (CosTaL 0.1) keeps scripts written for CosTaL 0.0.2 working: `CosTaL.find_communities`,
`CosTaL.clustering`, `CosTaL.pp` and `CosTaL.return_scores` keep their signatures and run on
CosTaL+. New code should use `costal_plus` directly. Please cite Li et al., *Briefings in
Bioinformatics* 2023, doi:10.1093/bib/bbad157.
