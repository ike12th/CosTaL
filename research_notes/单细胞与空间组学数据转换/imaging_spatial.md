# Count transformations / normalization for imaging-based single-cell spatial transcriptomics (Xenium / Xenium Prime 5K, MERFISH/MERSCOPE, CosMx 1K/6K/WTx), 2023–2026

Scope note: these notes cover cell-level count matrices from imaging-based platforms (im-SRT/iST). "Verified" items have an inline source. Items under **Inferences** are my reasoning and are not sourced claims. The CosTaL+ framing is: exact cosine kNN, then Tanimoto edge reweighting T = a·b / (‖a‖² + ‖b‖² − a·b), plus a choice between a sparse and a dense kNN backend.

Two facts drive most of the CosTaL+ conclusions below (both are inferences from the math, not from any source):
1. **Cosine is invariant to positive per-row scaling.** Cosine kNN graphs computed on raw counts, total-count-normalized counts (linear), cell-area/volume-normalized counts (linear), and XOA size-factor-normalized counts are therefore **identical**. The choice among these normalizations only changes the cosine graph once a nonlinear transform (log1p, sqrt) or a gene-wise transform (z-scoring, Pearson residuals, PCA) follows it.
2. **Tanimoto is not invariant to row scale.** For a fixed angle θ and norm ratio r = ‖a‖/‖b‖, T = r·cosθ / (r² + 1 − r·cosθ). This peaks at r = 1, so Tanimoto penalizes norm mismatch. That only helps if ‖row‖ means something biological in the chosen representation.

---

## Q1. What did Atta et al. (Genome Biology 2024) conclude about total-count vs cell-volume normalization and small/biased panels, and what came after (2025–2026)?

### Takeaway
Atta et al. showed that any **gene-count-derived** scaling factor (library size, scTransform, DESeq2, TMM) makes normalized expression depend on the panel. With region- or cell-type-skewed panels this creates false DE, flipped fold changes and missed SVGs. Cell-volume normalization (and no normalization) is immune because volume does not depend on which genes are in the panel. They recommend volume normalization first (area as a proxy), conditional on good segmentation. Follow-up work (Bhuva 2024, SpaNorm 2025) supports the related point that library size in spatial data carries real tissue-structure biology. I found **no 2025–2026 head-to-head benchmark of area vs total-count normalization specifically on Xenium/CosMx/MERSCOPE**.

### Cited Findings
- **Paper**: Lyla Atta, Kalen Clifton, Manjari Anant, Gohta Aihara, Jean Fan (Johns Hopkins), "Gene count normalization in single-cell imaging-based spatially resolved transcriptomics", Genome Biology, June 2024 (bioRxiv 2023.08.30.555624). DOI 10.1186/s13059-024-03303-w — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/); [DOAJ record](https://doaj.org/article/01f2c02da4014bcb949aa202a018ac3e). Vizgen also hosts it on its resources site — [Vizgen](https://vizgen.com/resources/gene-count-normalization-in-single-cell-imaging-based-spatially-resolved-transcriptomics/)
- **Datasets and simulated skewed panels**: MERFISH mouse brain (Vizgen Mouse Brain Receptor Map, 483 genes), with 100-gene panels skewed toward ventricle, habenula, fiber tracts or dentate gyrus. STARmap PLUS mouse brain (100-gene skewed). seqFISH mouse kidney (60-gene). CosMx human liver (200-gene, skewed to Zone 1). 10x Xenium human breast cancer (80-gene; DCIS vs invasive). Panel-size experiment on sorted-PBMC scRNA-seq with monocyte-skewed panels of 50, 100, 500, 1000 and 5000 genes — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Mechanism**: overrepresented genes inflate library-size scaling factors in the region or cell type where they are enriched. That pushes normalized expression **down** there and **up** elsewhere. Ventricle-cell scaling factors "tended to be systematically larger" — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Why volume is immune**: "as cell volume is independent of the number of gene counts detected in a cell, the scaling factor is the same with the skewed and full gene panels." Volume-normalized and unnormalized expression were perfectly correlated between skewed and full panels. In MERFISH, TMM ranged from about 0.5 to 0.98, library size and scTransform were similarly low, and DESeq2 stayed above 0.9 — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Magnitude of errors**:
  - DE: "up to a 13% error rate" in MERFISH, and "reaching up to 30% with some simulated gene panels" for library size and scTransform across datasets.
  - Fold-change direction switches: up to 19% in MERFISH and "up to 60%" across datasets, mostly for modest fold changes.
  - SVG false negatives: "reaching up to 24%" in CosMx.

  — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Panel size**: the effects persist with panels as large as 5000 genes but get smaller and less likely as panels grow. DESeq2 and TMM showed weaker region-specific effects than library size — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Implementation details**:
  - scTransform was evaluated via Pearson residuals with negatives set to 0.
  - Log10(x+1) was used before nnSVG and in a PCA simulation.
  - MERFISH provided 3D volume; seqFISH, CosMx and Xenium provided only 2D area.
  - "Cell area may serve as a proxy where volume estimates are not available", but the paper does not quantify how well area tracks volume.

  — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Verbatim recommendations**:
  - "We recommend using non-gene count-based normalization approaches when feasible."
  - "We recommend, as first choice, cell volume normalization to account for such partial cell capture within imaged regions."
  - Without volume data, even with skewed panels, "we still recommend utilizing a gene count-based normalization approach to account for variation in captured cell volume."
  - "the reliability of cell volume as a normalization factor is contingent upon the accuracy of cell segmentation."
  - "we caution against over-interpreting biological conclusions from effects with small magnitude gene expression changes."

  — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- Atta's workshop slides reportedly give a decision rule: use volume normalization if reliable volume exists; otherwise use count-based normalization if the panel is representative; if it is not, restrict analysis to specific regions or cell types. This comes from a search-engine summary of the slides; I did not read the slides directly — [ISB slides](https://isbscience.org/spatial-multiomics-workshop/assets/slides/Atta.pdf)
- **Bhuva et al., "Library size confounds biology in spatial transcriptomics data"**, Genome Biology 2024 (25:99). It used 25 samples from Visium, Xenium, CosMx and STOmics. Library size differed significantly across tissue structures regardless of technology, and correcting it (incl. sctransform) worsened spatial-domain identification. The authors say spatial data "should not be specifically corrected for library size prior to analysis" — [DOAJ](https://doaj.org/article/b13da81ba36848aea93f840263cbb1cf); [GenomeWeb coverage](https://www.genomeweb.com/node/624226); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.15.532733.full.pdf)
- **SpaNorm** (Genome Biology 2025) is a spatially-aware normalization that models region-varying library size. It beat common single-cell normalizations across 27 samples, 6 datasets and 4 platforms. Plain library-size normalization was best in 7 samples, 6 of which were Visium — [SpaNorm](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03565-y)
- **GPSNorm** (bioRxiv, July 2026) is a Bayesian spatial normalization folded into inference. It was validated on Visium, GeoMx and kidney data, not on Xenium/CosMx — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.23.740319v1)
- **Salas et al., Nature Methods 2025** (Xenium benchmark; details in Q4) found that "the most essential factors in the processing workflow were the normalization method", along with library size, scaling and #PCs. They also say "there is no universal best way to process spatial datasets." Area normalization was **not** in their cell-level benchmark — [PMC11978515](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978515/)

### Inferences
- The Atta, Bhuva and Danaher (Q2) results converge: in im-SRT, per-cell total count = detection efficiency × captured volume × true RNA content × panel composition × segmentation error. Total-count normalization removes all five indiscriminately. Area/volume normalization removes only the volume term, and only as well as the segmentation estimates it.
- The Atta errors shrink with panel size but do not vanish at 5000 genes. Xenium Prime 5K and CosMx 6K are therefore much less exposed than 300–500-gene marker panels, but not immune. WTx (~19K) should behave close to scRNA-seq in this respect.

### Gaps
- No 2025–2026 study benchmarks area/volume vs total-count normalization head-to-head on real Xenium, CosMx or MERSCOPE data with ground truth. Searches for one came back empty.
- How well 2D area tracks 3D volume in 5 µm FFPE sections was not quantified in any source I found.

---

## Q2. Default processing in vendor pipelines and community tutorials (quotes)

### Takeaway
Vendors ship **raw integer counts** plus per-cell geometry (area/volume), and their onboard secondary analysis uses library-size-type scaling, then PCA and graph clustering. Bruker explicitly recommends total-count normalization on the **linear** scale, with log/sqrt/Pearson residuals reserved for UMAP and clustering. Community tutorials split three ways:
- Scanpy/Squidpy: normalize_total + log1p + PCA.
- Seurat: SCTransform (clipped Pearson residuals) + PCA.
- Bioconductor OSTA: **area-based size factors** + log + PCA.

Vizgen publishes volume in cell metadata but I found no official Vizgen normalization recommendation.

### Cited Findings
**10x Xenium Onboard Analysis (XOA) / Xenium Ranger**
- Secondary analysis uses only Q≥20 transcripts assigned to cells and drops cells with fewer than 5 transcripts. PCA reduces cells × genes to cells × 10 PCs using a modified IRLBA. Two clustering methods (graph-based Louvain and k-means) both operate "in the PCA space". UMAP defaults: n_neighbors=30, min_dist=0.3, metric="pearson" — [10x XOA secondary analysis docs](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/algorithms-overview/gex-secondary-analysis-algorithms) (read via search snippets; direct fetch was rate-limited)
- Size factor: "The size factor for each cell is the total transcript count in that cell divided by the median transcript count per cell (across all cells)." This differs from sSeq in using total counts rather than DESeq's geometric mean. It is used for per-cluster mean expression and DE — [10x XOA docs](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/algorithms-overview/gex-secondary-analysis-algorithms)
- XOA reuses Cell Ranger algorithms. It merges clusters by hierarchical clustering of medoids in PCA space, merging siblings with no DE genes (BH-adjusted p<0.05) — [10x docs via search](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/algorithms-overview/gex-secondary-analysis-algorithms)
- The `analysis/` directory holds graph-based and k-means clusterings, DE, PCA (components, dispersion, features_selected, projection, variance) and UMAP — [Xenium file format docs](https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html)
- "Median transcripts per cell" excludes zero-transcript cells and "will depend on tissue type and the selected gene panel" — [XOA analysis summary](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-analysis-summary)

**Vizgen MERSCOPE (vpt / Vizualizer)**
- The vpt segmentation suite outputs a cell-by-gene file, a cell metadata file (which includes volume) and QC metrics — [Galaxy vpt news](https://archive.galaxyproject.org/news/2025-05-19-vpt/); [vpt on PyPI](https://pypi.org/project/vpt/); [spatialdata-io merscope reader](https://spatialdata.scverse.org/projects/io/en/stable/generated/spatialdata_io.merscope.html)
- The merfish-tools notebook (academic, epigen-UCSD) for MERSCOPE output does:
  - `sc.pp.normalize_total(adata)` and `sc.pp.log1p(adata, base=2)`, with no volume normalization.
  - It states PCA is "not required" and builds the kNN with `sc.pp.neighbors(adata, use_rep="X", n_neighbors=30, metric="cosine")`. This is a direct precedent for **cosine kNN on the full log-normalized gene space** of a ~500-gene panel.

  — [merfish-tools](https://merfish-tools.readthedocs.io/en/latest/notebooks/analyze_merscope_experiment.html)

**Bruker/NanoString CosMx, AtoMx SIP (v1.3.2 guidance)**
- Normalization options:
  - Total Counts is "recommended": "a global-scaling normalization method that normalizes gene counts for each cell by the total expression of each cell."
  - Seurat normalization is "total counts normalization that is multiplied by a scale factor (10,000) and natural-log transformed after adding 1."
  - Pearson residuals "can outperform other normalization methods" in datasets of ≤500,000 cells; for larger ones "we typically recommend a total counts normalization."

  — [Bruker blog](https://brukerspatialbiology.com/blog/tips-when-performing-cosmx-data-analysis-with-atomx-sip/)
- "For differential expression and many advanced analysis modules, unnormalized data is utilized." After normalization, "run the PCA … with 50 PCs." UMAP defaults: 30 neighbors, min dist 0.01, spread 5, **Distance Metric: Cosine** — [Bruker blog](https://brukerspatialbiology.com/blog/tips-when-performing-cosmx-data-analysis-with-atomx-sip/)
- **CosMx Analysis Scratch Space** (Patrick Danaher, Bruker; Jan 2024, modified Jun 2024), "QC and normalization of RNA data":
  - It assumes "a cell's detection efficiency is well-estimated by its total counts" and divides each cell by its total counts, with a floor of 20 that "puts a floor on how much we'll up-scale a cell."
  - It concedes normalization discards biology, since "cancer cells tend to have high overall RNA expression", but calls this "a small price to pay to control the variability in total counts."
  - "We recommend keeping data on the linear scale."
  - Log/sqrt/log1p "inflate the variability of low counts and shrink the variability of high counts" and help UMAP and Leiden/Louvain clustering.
  - Pearson residuals "can produce great UMAPs" but are "only viable in small studies" because they create a dense matrix with a large memory footprint.

  — [Scratch Space normalization post](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/index.html)
- Bruker released **scPearsonPCA**, an R package for efficient PCA of Pearson-residual-normalized data. This is engineering around the density problem — [Scratch Space index](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/)
- A community pipeline built on Scratch Space material reports that the basic vignette does total-count normalization, then a sqrt transform for UMAP. Raw counts are used for cell typing (InSituType) and normalized, untransformed counts for neighborhood analysis. This is a secondary source — [LGA CosMx pipeline](https://rawcdn.githack.com/lga-uoc/CosMx_pipeline_LGA/fe7b35df54070027de122384e9310111affd1491/docs/2.0_normalization.html)

**Seurat v5 "Analysis of Image-based Spatial Data"**
- Vizgen: `SCTransform(vizgen.obj, assay="Vizgen", clip.range=c(-10,10))`, then `RunPCA(npcs=30, features=rownames(obj))`, `FindNeighbors(dims=1:30)`, `FindClusters(resolution=0.3)`. The clipping is changed "to mitigate the effect of outliers" in smFISH data.
- Xenium: `SCTransform(xenium.obj, assay="Xenium")` with default clipping, then 30 PCs.
- CosMx: `SCTransform(..., clip.range=c(-10,10))`, then mapping to an Azimuth reference.

  — [Seurat vignette](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2)

**Squidpy (scverse)**
- Xenium tutorial (480-gene panel):
  - `filter_cells(min_counts=10)` and `filter_genes(min_cells=5)`.
  - `normalize_total` with no `target_sum`, so the default applies.
  - `log1p`, then `pca`, `neighbors`, `umap` and `leiden` with defaults and no scaling.

  — [Squidpy Xenium](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_xenium.html)
- Vizgen tutorial (483-gene MERFISH): same normalize_total + log1p + PCA + Leiden flow, with HVG `seurat_v3` n_top_genes=4000 run on counts. It states "Alternatively, counts could also be normalized by volume" but does not do it — [Squidpy Vizgen](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_vizgen.html)

**Giotto**
- Default `normalizeGiotto()` "standard" method:
  1. Library-size normalization × scalefactor (default 6000).
  2. log2(x+1).
  3. Z-scoring by features, then by cells.
- Alternatives: `pearson_resid` (stored in the "scaled" slot; "scaling is not recommended after this"; not for DE) and an "osmFISH" method.

  — [Giotto normalization article](https://giottosuite.com/articles/normalization.html); [normalizeGiotto reference](https://giottosuite.com/reference/normalizeGiotto.html)

**Bioconductor OSTA book (Orchestrating Spatial Transcriptomics Analysis)**
- Xenium workflow (313/422-plex): "Next, we'll log-normalize counts by area, and perform principal component analysis (PCA) on all 422 RNA targets." Size factors are `cell_area / median(cell_area)`. Clustering is SNN (Jaccard) graph plus Leiden — [OSTA Xenium](https://lmweber.org/OSTA/pages/img-workflow-xenium.html)
- CosMx workflow (950-plex mouse brain): the same area-based size factors (`Area/median(Area)`), log-normalization and PCA on all targets — [OSTA CosMx](https://lmweber.org/OSTA/pages/img-workflow-cosmx.html)

**Independent papers' processing**
- Ren et al. (Nat Commun 2025; Xenium 5K vs CosMx 6K): "Data were normalized and log-transformed to the same scale"; top 10% variance genes as HVGs; 30 PCs; Leiden (scanpy 1.10.3) — [PMC12534522](https://pmc.ncbi.nlm.nih.gov/articles/PMC12534522/)
- The Nature Biotechnology 2025 imaging-ST metrics paper reportedly used SCTransform (defaults) for normalization and feature selection/visualization and **raw counts** for cell-type annotation (from a search summary; full text not read) — [Nat Biotech 2025](https://www.nature.com/articles/s41587-025-02811-9)

### Inferences
- The de facto community default for Xenium and MERSCOPE is still scRNA-seq-style **normalize_total → log1p → (scale) → PCA → kNN (Euclidean) → Leiden**. Area normalization is a minority practice, most visibly in OSTA (Bioconductor), despite Atta et al.
- Both vendors' own embeddings use **correlation/cosine-type metrics**: XOA UMAP uses "pearson" and AtoMx UMAP uses "Cosine". merfish-tools builds a cosine kNN without PCA. That supports CosTaL+'s cosine choice for small imaging panels.
- XOA's documented size factor is median-scaled total counts, which is a pure row scaling. The cosine geometry of XOA-normalized data before any log is therefore identical to raw counts.

### Gaps
- I could not confirm from the 10x docs whether XOA log-transforms before PCA (direct page fetch hit HTTP 429), or how Xenium Explorer displays expression (raw vs normalized).
- I found no official Vizgen documentation (MERSCOPE Vizualizer, vpt) stating a recommended normalization. vpt outputs raw counts plus volume.
- The AtoMx docs I read do not state which normalization is pre-selected in the UI.

---

## Q3. Effects of cell segmentation (vendor default, Baysor, Proseg, Cellpose, nuclear expansion) on counts per cell and on normalization

### Takeaway
Segmentation sets both the numerator (assigned transcripts) and the denominator (area/volume) of any normalization.
- Vendor defaults leave about 20% or more of transcripts unassigned.
- Nuclear expansion inflates area and imports neighbors' transcripts.
- Baysor raises counts per cell but can over-segment.
- Proseg reduces spurious coexpression.

Misassigned molecules from neighbors are now shown to dominate DE and ligand–receptor results (Mitchel et al., Nat Genet 2026). Contamination is additive, not multiplicative, so **no per-cell scalar normalization (total-count or area) can remove it**.

### Cited Findings
- **Salas et al. 2025 (Nature Methods, 25 Xenium datasets)**:
  - Default Xenium segmentation averaged "186.6 reads per cell", with "76.8% of reads being assigned to cells", on 210–392-gene panels.
  - "Cell mask expansion after segmentation can negatively impact the characterization of cell populations."
  - Baysor + Xenium nuclei prior (BA2 P0.8) was the best strategy by reads assigned and negative marker purity.
  - "Although cells defined by Baysor had a higher count per cell, the identified cellular populations were the same."
  - "Xenium's default nuclear segmentation masks provide adequate information for defining the main populations."

  — [PMC11978515](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978515/)
- **Proseg** (Jones et al., "Cell simulation as cell segmentation", Nature Methods 2025, 22:1331–1342):
  - Assignment: default image-based methods leave "20% or more of transcripts" unassigned, and MERSCOPE's provided Cellpose "failed to assign roughly 26%". Proseg and Baysor assign nearly all transcripts.
  - Cell counts: Baysor produced "nearly 4 million" CosMx cells vs about 700K (Proseg) and 783K (provided Cellpose). On MERSCOPE, Baysor added 230K cells (62% more than Proseg).
  - Median transcripts per cell:
    - MERSCOPE (provided segmentation): T cells 81, neutrophils 96, tumor 291.
    - CosMx (Proseg): B cells 183, neutrophils 69, fibroblasts 271, tumor 405.
    - Xenium RCC (Proseg): T cells 64, macrophages 180, tumor 226.
  - Cellpose and Baysor cut neutrophil/T-cell proportions "by nearly half" on MERSCOPE, because small cells get "smothered by adjacent higher population cells".
  - Xenium lung DE: "With the provided image-based segmentation 37% of calls were suspicious, compared to 12% with Proseg and 25% with Baysor" (nuclear-only 28%).
  - The Xenium default nuclear expansion was 15 µm; "newer versions of the platform have reduced the default radius to 5μm."
  - Proseg's per-cell rates are "interpretable as the expected gene expression, normalized by cell volume."

  — [Proseg preprint, PMC11071468](https://pmc.ncbi.nlm.nih.gov/articles/PMC11071468/)
- Proseg 3 output: "By default, count matrices generated by Proseg 3 are integer point estimates." `--output-expected-counts` gives "non-integer estimates from taking the mean over multiple samples". Cell metadata includes volume. The method is non-deterministic: "results will vary slightly" — [docs.rs proseg 3.2.0](https://docs.rs/proseg)
- **Mitchel, Gao, Petukhov, Cole, Kharchenko**, "Impact and correction of segmentation errors in spatial transcriptomics", Nature Genetics 58:434–444 (Jan 2026):
  - Segmentation errors "confound most downstream analysis of cellular state" (DE, neighbor influence, ligand–receptor), with impact "frequently dominating the results".
  - Proposed fix: factorization of local molecular neighborhoods (the cellAdmix tool), analogous to doublet filtering.

  — [bioRxiv 2025.01.02.631135](https://www.biorxiv.org/content/10.1101/2025.01.02.631135.full.pdf); [Broad listing](https://www.broadinstitute.org/publications/broad1373251)
- **Bruker (Danaher, Jan 2024)**: cells are "contaminated with transcripts properly belonging to other cells." Spurious hits such as keratins in tumor-bed T cells "are often the most significant genes emerging from a DE analysis." Suggested mitigations: filter genes with much higher expression around a cell type than inside it, or add a neighbor-expression covariate, which "only ameliorates, not solves" — [Scratch Space segmentation-error post](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/segmentation-error/index.html)
- **Nature Methods 2026** (Gottardo/Homicsko, CHUV/UNIL), "Resolving sensitivity, specificity and signal contamination in Xenium spatial transcriptomics": transcript dispersion assigns signal to neighboring cells, producing "artificially mixed" profiles that can be misread as interactions or hybrid states. They used snRNA-seq references to quantify contamination and compare panel designs. Details come from a press release; full text not read — [CHUV news](https://www.chuv.ch/en/oncologie/onc-home/liste-des-actualites/detail/news/64397-nouvelle-methode-pour-affiner-la-transcriptomique-spatiale-tumorale); [Nature Index listing](https://www.nature.com/nature-index/article/10.1038/s41592-026-03089-8)
- **MisTIC** (bioRxiv, Dec 2025) is a variational Bayesian correction of transcript misassignment without re-segmentation. It notes that errors appear as co-expression of mutually exclusive genes — [PMC12724677](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12724677/)
- **Ozirmak Lermi et al.** (Nat Commun 2025; CosMx vs MERFISH vs Xenium on FFPE lung/mesothelioma TMAs): "cell area sizes were crucial for interpreting transcript counts per cell." A Xenium segmentation setting (unimodal/nuclear expansion) produced larger cell areas than morphology-marker-guided segmentation (CosMx, MERFISH, Xenium-MM). They normalized per-cell transcript and gene counts by panel size for cross-platform comparison. This comes from search snippets; full text not read — [Nat Commun 2025](https://www.nature.com/articles/s41467-025-63414-1)
- Atta et al.: volume normalization's reliability "is contingent upon the accuracy of cell segmentation" — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- Hartman & Satija (eLife reviewed preprint): counts per cell "are not directly comparable across datasets" because off-target molecular artifacts differ in incidence. Molecular false positives distort spatial DE — [eLife RP 96949](https://elifesciences.org/reviewed-preprints/96949)

### Inferences
- Segmentation changes the denominator of area/volume normalization (expansion inflates area) and the numerator (assigned counts) in correlated but non-proportional ways. Small cells adjacent to large, transcript-dense cells (T cells, neutrophils next to tumor) are hit hardest. Their raw L2 norm is dominated by imported tumor transcripts, which inflates their similarity to tumor in both cosine and Tanimoto.
- Contamination adds a neighbor profile to a cell's vector, so it rotates the vector (changes cosine) and changes its norm. Row scaling cannot undo this. For CosTaL+, this means:
  - Tanimoto's norm term on raw or area-normalized data will partly encode segmentation artifact (mask size, neighbor density), not just biology.
  - For Proseg expected-count output, matrices are non-integer and, by construction (averaging posterior samples), likely to have fewer exact zeros than integer outputs. That would matter for the sparse backend; I did not verify this numerically.

### Gaps
- No source quantified the typical fraction of a cell's counts that are contamination per platform. The Mitchel numbers are only in the paper body, which I could not read.
- I did not find quantitative data on how much Xenium's newer multimodal segmentation (boundary/interior stains, 5 µm expansion) changes median counts per cell vs the old 15 µm expansion on the same tissue.

---

## Q4. Are log1p, Pearson residuals, and area scaling common? Do people cluster on PCA?

### Takeaway
Yes, PCA-based clustering is the norm.
- **log1p after normalize_total** is the most common (Scanpy/Squidpy, Ren 2025, Salas 2025's best workflow).
- **Pearson residuals via SCTransform** are standard in Seurat vignettes for Xenium/Vizgen/CosMx and are an AtoMx option, but Bruker calls them memory-heavy, small-study-only and dense.
- **Area scaling** appears in OSTA and in Atta's recommendation but is not the default anywhere I found.
- Typical PC counts: 10 (XOA), 30 (Seurat, Ren), 40 (Salas main analysis), 50 (AtoMx). Some workflows skip PCA entirely for small panels (merfish-tools cosine kNN on X; Salas used "all principal components").

### Cited Findings
- Salas et al. 2025 benchmarked:
  - Normalization: Pearson residuals, SCTransform, library-size normalization (target 10, 100, 1,000, or default), or none.
  - Optional log, HVF selection and scaling.
  - #PCs and #neighbors.
  - Leiden, Louvain or MCL clustering.

  The best: "(1) library-size-based normalization, with the total library size set to 100; (2) log-transformation; (3) scaling; (4) the construction of a k-nearest neighbors graph using all principal components and 16 neighbors; and (5) Louvain clustering." Some top workflows skipped log but kept scaling. Main-population analysis used 40 PCs and 12 neighbors with Leiden — [PMC11978515](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978515/)
- Seurat uses SCTransform for Xenium, Vizgen and CosMx, with clip.range ±10 for smFISH-type data, and 30 PCs — [Seurat vignette](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2)
- AtoMx offers Pearson residuals for ≤500K cells and total counts above that, with 50 PCs — [Bruker blog](https://brukerspatialbiology.com/blog/tips-when-performing-cosmx-data-analysis-with-atomx-sip/)
- Bruker recommends the linear scale, with log/sqrt only for UMAP/Leiden. Pearson residuals produce a dense, memory-heavy matrix — [Scratch Space](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/index.html)
- Giotto's default includes z-scoring (dense) after log2 — [Giotto](https://giottosuite.com/articles/normalization.html)
- OSTA uses area-based size factors plus log for both Xenium and CosMx — [OSTA Xenium](https://lmweber.org/OSTA/pages/img-workflow-xenium.html); [OSTA CosMx](https://lmweber.org/OSTA/pages/img-workflow-cosmx.html)
- XOA clusters in a 10-PC space — [10x XOA docs](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/algorithms-overview/gex-secondary-analysis-algorithms)
- merfish-tools: "it is not required to use PCA for feature reduction"; kNN with `use_rep="X", metric="cosine"` — [merfish-tools](https://merfish-tools.readthedocs.io/en/latest/notebooks/analyze_merscope_experiment.html)
- Atta et al. note that scTransform "does not use cell-specific scaling factors" (it regresses library size within a GLM), so it was omitted from their scaling-factor analysis (via search summary of the paper) — [Genome Biology](https://link.springer.com/article/10.1186/s13059-024-03303-w)

### Inferences
- Salas's best "library size = 100" is notable: with ~190 counts per cell, scaling to 100 keeps most values in a range where log1p is only mildly compressive. That workflow is closer to "scaled linear proportions" than to scRNA-seq-style CP10k log1p.
- For 300–500-gene panels, PCA is optional. Several practitioners and the vendors' own UMAPs work directly in gene space with cosine or Pearson metrics. CosTaL+'s exact cosine kNN on the full (non-PCA) matrix fits that practice.

### Gaps
- No survey quantifies how often published Xenium/CosMx/MERSCOPE papers use each normalization. "Most common" above is inferred from tutorials and vendor docs.
- The scanpy `normalize_total` default (target = median total count when `target_sum=None`) is from my knowledge of the scanpy docstring; I did not re-fetch it this session.

---

## Q5. Marker-designed (non-random) panels: what goes wrong with total-count normalization?

### Takeaway
In a marker panel, a cell's total count is dominated by whichever panel markers its type expresses. Total-count normalization therefore turns panel design into a cell-type- and region-specific scaling artifact: it depresses every gene in "marker-rich" cells and inflates them in "marker-poor" cells. The results are false DE, flipped fold changes and lost SVGs (Atta). It also removes real biology, namely RNA content and density differences tied to tissue structure (Bhuva, Danaher).

### Cited Findings
- Overrepresented region-specific genes produce systematically larger scaling factors there, pushing normalized expression down in that region and up elsewhere. The consequences:
  - DE errors up to 13% (MERFISH) and up to 30% (library size/scTransform across datasets).
  - FC switches up to 60%.
  - SVG false negatives up to 24% (CosMx).

  — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- Library size and scTransform assumptions are violated by marker-based panels. DESeq2 and TMM are less affected but carry their own assumptions — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- Effects persist at 5000 genes but shrink with panel size — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- Library size varies with tissue structure on Xenium and CosMx; correcting it harms domain identification — [GenomeWeb on Bhuva 2024](https://www.genomeweb.com/node/624226)
- Bruker acknowledges that total-count normalization removes real signal ("cancer cells tend to have high overall RNA expression") and accepts the trade-off — [Scratch Space](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/index.html)

### Inferences
- **Compositional closure.** With total-count normalization, each row lies on a simplex (L1 = const). In a 300-gene panel where, say, tumor cells spend most counts on a handful of keratin/epithelial markers, every other gene's normalized value in tumor cells is mechanically suppressed. This is the same closure problem as in compositional data.
- **Effect on cosine/Tanimoto.**
  - The cosine graph is unaffected by the linear normalization itself (row scaling), so the Atta artifact enters CosTaL+'s cosine graph only through the log1p that typically follows.
  - log1p(x·s) with a cell-specific s is not a rotation-preserving map. Cells with large s (low totals) get expanded nonzero entries and different angles.
  - For Tanimoto on linear total-count-normalized rows, ‖a‖₂ = S·sqrt(Σp_i²), a Simpson concentration index. In a marker panel it reflects how concentrated a cell's counts are on a few panel markers. That is biologically structured but panel-design-dependent.
- **Area normalization** avoids the panel-composition artifact, but its norm (transcript density per µm²) inherits segmentation error and section-thickness and partial-capture effects (Q3).

### Gaps
- No study I found measured how marker-panel total-count normalization affects **clustering/kNN graphs** specifically, as opposed to DE, FC and SVG.

---

## Q6. Output-matrix properties per transformation: (a) sparse/dense, typical counts and zero fraction; (b) can be negative; (c) does row norm carry biology

### Takeaway
Only nonnegative, zero-preserving row-scalings and elementwise transforms keep the data sparse: raw, total-count, area/volume, size-factor, log1p, sqrt. Gene-centering transforms (z-score, Pearson residuals, PCA) produce dense matrices with negative values. In those, the norm means "distance from the average cell", not "amount of RNA".
- Magnitude carries the most biology in **raw counts** and **area/volume-normalized counts** (RNA content/density), but these are also the most contaminated by technical and segmentation variation.
- After **total-count normalization**, the L1 norm is constant by construction and the L2 norm only encodes expression concentration.
- Typical sparsity: roughly 60–90% zeros for 300–500-gene panels and roughly 85–97% for 5K/6K/WTx panels (estimates; see inferences).

### Cited Findings: per-cell counts (inputs to sparsity estimates)
- **Xenium v1 (210–392-gene panels)**: average 186.6 reads per cell, 76.8% of reads assigned — [Salas 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978515/)
- **Xenium RCC (Proseg)**: medians of 64 (T), 180 (macrophage) and 226 (tumor) transcripts per cell — [Proseg](https://pmc.ncbi.nlm.nih.gov/articles/PMC11071468/)
- **MERSCOPE (provided segmentation)**: medians of 81 (T), 96 (neutrophil) and 291 (tumor) — [Proseg](https://pmc.ncbi.nlm.nih.gov/articles/PMC11071468/)
- **CosMx (~1K panel, Proseg)**: medians of 69 to 405 by cell type — [Proseg](https://pmc.ncbi.nlm.nih.gov/articles/PMC11071468/)
- **Xenium Prime 5K**: median transcripts per cell from 10x datasets:
  - 1,401 in fresh-frozen human ovarian adenocarcinoma (XOA 3.0.0) — [10x dataset](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-human-ovary)
  - 306 in FFPE melanoma (dev panel, 5,006 genes) — [10x dataset](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin)
  - 144 in FFPE prostate (dev panel) — [10x dataset](https://10xgenomics.com/datasets/xenium-prime-ffpe-human-prostate)
  - 255 in FFPE lymph node preview (4,624-gene dev panel) — [10x dataset](https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression)

  These medians were read via search snippets. Median genes per cell was not found.
- **CosMx 6K Discovery (FFPE lymph node)**: median 680 / mean 744 transcripts per cell, mean 461 unique genes per cell, about 1.85M cells — [NanoString dataset page](https://nanostring.com/wp-content/uploads/2024/11/MK6578_CosMx_Human_Lymph_Node_FFPE_Dataset_webpage.html)
- **CosMx 6K vs WTx on serial sections**: medians of 1,567 transcripts / 926 genes (6K) vs 2,458 transcripts / 1,677 genes (WTx). This is a vendor poster — [AACR 2025 poster](https://nanostring.com/wp-content/uploads/2025/05/2779-AACR-2025_WTX_Cell_Assay_MerckBruker_Yi-Cui_MSedits_HS.pdf)
- **CosMx WTx (colorectal FFPE)**: median 967 transcripts and 627 genes per cell, vs scRNA-seq 1,151 and 774. Vendor blog — [Bruker blog](https://brukerspatialbiology.com/blog/moving-beyond-scrna-seq-the-power-of-cosmx-whole-transcriptome/)
- **Ren et al. 2025** (Xenium 5K vs CosMx 6K, colon/HCC/ovarian): "CosMx 6K detected a higher total number of transcripts than Xenium 5K." On 2,522 shared genes, both platforms had "comparable transcripts and genes per cell to scRNA-seq." Exact per-cell values are only in figures and source data — [PMC12534522](https://pmc.ncbi.nlm.nih.gov/articles/PMC12534522/)
- **Pearson residuals** produce a dense matrix with a large memory footprint — [Bruker Scratch Space](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/index.html)
- **Seurat clips SCT residuals** at ±10 for Vizgen/CosMx, which confirms negative values in the output — [Seurat](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2)
- **Atta set negative scTransform residuals to 0** for their analysis — [PMC11167774](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)
- **Giotto's default** z-scores by genes and cells (produces negatives) — [Giotto](https://giottosuite.com/articles/normalization.html)
- **Proseg 3** gives integer counts by default; optional expected counts are non-integer — [docs.rs](https://docs.rs/proseg)
- **Magnitude as biology**:
  - Library size tracks tissue structure — [Bhuva via GenomeWeb](https://www.genomeweb.com/node/624226)
  - Cancer cells have higher overall RNA — [Danaher](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/index.html)
  - Cell area is "crucial for interpreting transcript counts per cell" — [Ozirmak Lermi 2025](https://www.nature.com/articles/s41467-025-63414-1)
  - Partial-volume capture is a technical component of counts — [Atta](https://pmc.ncbi.nlm.nih.gov/articles/PMC11167774/)

### Inferences: property table (my analysis; sparsity numbers are estimates)

Zero-fraction estimate: zero fraction = 1 − (genes detected / panel size). The hard lower bound is 1 − min(counts, panel)/panel.

| Transformation (where used) | (a) Sparse? | (b) Negative? | (c) Row norm carries biology? | Cosine graph vs raw | Tanimoto adds over cosine? |
|---|---|---|---|---|---|
| Raw counts (XOA/vpt/AtoMx export; AtoMx DE; InSituType) | Sparse. Small panels (~100–400 counts/cell over 300–500 genes): likely ~60–85% zeros. 5K/6K: CosMx 6K LN 461 genes of ~6K means ~92% zeros; 6K poster 926 genes means ~85%; Xenium 5K at 144–1,401 counts over ~5K genes means **at least** 72–97% zeros by the hard bound alone. WTx: 627–1,677 of ~19K genes means ~91–97% zeros | No (≥0 integers) | **Yes, mixed**: total RNA content (biology: tumor vs immune) × captured volume × detection efficiency × segmentation/contamination. Highest information and highest confounding | Same as all row-scalings | Yes. It penalizes total-count mismatch, which partly reflects cell size or type and partly artifact |
| Total-count / library-size, linear (AtoMx "Total Counts", Bruker recommended; XOA size factors; normalize_total without log) | Same zeros as raw | No | L1 is constant by construction. L2 = S·sqrt(Σp²) encodes only **concentration** (how marker-dominated a cell is). This is panel-design-dependent | **Identical** to raw (cosine is scale-invariant) | Weakly. It only separates cells with similar angle but different concentration |
| Area/volume normalization, linear (Atta's first choice; Squidpy "alternatively") | Same zeros as raw | No | **Yes**: transcript density per µm²/µm³. Biological (RNA density) plus segmentation and section-thickness artifacts | **Identical** to raw | Yes, if segmentation is trustworthy |
| log1p(total-count × target) (Scanpy/Squidpy, AtoMx "Seurat", Ren 2025, Salas best with target=100) | Sparse (log1p(0)=0), same zeros | No | Partly. The norm grows with number of detected genes and with the target/total ratio (low-count cells get inflated nonzeros). Mostly a depth/complexity proxy, compressed | **Differs** from raw (nonlinear) | Marginal and depth-confounded |
| log(area-normalized + 1) (OSTA Xenium/CosMx) | Sparse | No | Yes, compressed density | Differs | Plausibly useful (density) |
| sqrt of normalized counts (Bruker vignette for UMAP) | Sparse | No | Compressed version of the input's norm | Differs | As above for its input |
| Gene z-scoring / scaling (sc.pp.scale, Seurat ScaleData, Giotto default, Salas best) | **Dense** (zeros become −mean/sd) | **Yes** | The norm is distance from the average profile in standardized units. Rare or extreme cells have large norms; not "amount of RNA" | Differs (cosine about the mean is closer to correlation) | Questionable. The norm measures atypicality, and centering makes a·b sign-indefinite |
| Pearson residuals / SCTransform (Seurat vignettes, AtoMx option, Giotto `pearson_resid`, Salas candidates) | **Dense**; Bruker: memory-heavy | **Yes**; clipped to ±10 in Seurat smFISH settings, or ±sqrt(n) by default | The norm reflects deviation from depth-expected expression. Low-depth cells have small negative residuals for zeros and large positive ones for detected genes, so the norm is still depth-entangled | Differs | Questionable (sign-indefinite; T can be ≤0) |
| PCA scores (XOA 10 PCs; Seurat 30; AtoMx 50; Salas all PCs) | **Dense** (n × k) | **Yes** | The norm is distance from the centroid in PC space | Differs | Questionable, as for z-scores |
| Proseg expected counts (`--output-expected-counts`) | Probably less sparse than integer output (posterior means); not verified | No | Volume-aware model; the norm is similar to raw counts | n/a | n/a |

Additional inferences for CosTaL+:
- **Backend choice.**
  - For 300–500-gene panels, a dense n×G float32 matrix is small (G ≤ 500). With a zero fraction of perhaps 60–85%, sparse storage gives limited speed or memory benefit, so a dense backend is likely competitive or faster.
  - For 5K/6K/WTx with 85–97% zeros and G = 5,000–19,000, sparse storage clearly wins for zero-preserving transforms.
  - Any centering transform (z-score, Pearson residuals, PCA) forces a dense backend. For 5K/6K that means PCA first (as Bruker's scPearsonPCA does) rather than a dense G-dimensional matrix.
- **Where Tanimoto can add value** (magnitude is informative): raw counts and area/volume-normalized counts, linear or log. There the norm reflects RNA content or density, which differs between cell types (e.g., tumor vs lymphocyte). The caveat is that it is entangled with segmentation and contamination (Q3) and detection efficiency (Atta).
- **Where Tanimoto adds little**: after total-count normalization (L1 fixed), and after log1p of CP10k-style data (the norm is mostly a depth/complexity proxy).
- **Where Tanimoto is ill-posed**: on centered, sign-indefinite representations (z-scores, Pearson residuals, PCA). There a·b can be negative, so T can be negative or non-monotone in cosine.
- **Practical note**: because cosine is row-scale-invariant, "raw vs total-count vs area normalization" is a no-op for the cosine kNN stage. It matters only for the Tanimoto reweighting and for any nonlinear transform applied afterward. CosTaL+ can therefore expose normalization choice as purely a "what does magnitude mean" decision for the Tanimoto step.

### Gaps
- I found no published zero-fraction (or median genes per cell) values for Xenium v1 small panels, MERSCOPE 500-gene panels or Xenium Prime 5K. The sparsity ranges above are bounded estimates from transcripts per cell, panel size, and (for CosMx) genes per cell. Computing exact numbers would need the public 10x/NanoString/Vizgen datasets.
- I found no study testing whether per-cell norm (total counts or density) improves cell-type kNN or clustering accuracy in imaging ST, i.e., direct evidence for or against Tanimoto-style magnitude weighting.
