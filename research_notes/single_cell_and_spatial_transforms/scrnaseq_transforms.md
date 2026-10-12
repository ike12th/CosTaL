# scRNA-seq count transformations in 2023–2026 practice: defaults, benchmarks, and the sparsity, sign and row-norm properties of their output

Scope: what transformations are standard now (Scanpy/scverse, Seurat v5, Bioconductor/OSCA), what Ahlmann-Eltze & Huber 2023 and later benchmarks conclude, whether CosTaL's "log1p on raw counts, no size factors" is acceptable, what matrix kNN graphs are built on, and for every transformation three properties of the output: (a) sparse? (b) can be negative? (c) does the per-cell row norm carry biology, or is it technical / normalized away?

Why (c) matters for CosTaL+: Tanimoto similarity can be written as a function of cosine and the norm ratio only. With r = ‖a‖/‖b‖:
T = ‖a‖‖b‖cos / (‖a‖² + ‖b‖² − ‖a‖‖b‖cos) = cos / (r + 1/r − cos).
So T adds information over cosine only through r. For a fixed cosine it is largest when r = 1 and falls as the norms diverge. Whatever drives the row norm (sequencing depth, profile concentration, distance from the dataset centroid) is exactly what the Tanimoto reweighting penalizes. For non-negative vectors T ∈ [0, 1]. For signed vectors T ∈ [−1/3, 1] (at r = 1 and cos = −1, T = −1/3). This is my own algebra, not taken from a source.

Local check of the original CosTaL preprocessing: `bench/reference/CosTaL-0.0.2.4/CosTaL/clustering.py` lines 98–103 run `filter_cells(min_genes=200)`, `filter_genes(min_cells=3)`, `highly_variable_genes(n_top_genes=2000, flavor='seurat_v3')`, then `log1p`, then subset to HVGs. There is no `normalize_total` and no PCA. The kNN is computed on the sparse 2000-gene log1p matrix. `docs/06_cross_platform_and_determinism.md` (§5.1) records this as the "high-dimensional sparse" input class served by the L2Knng-style inverted index.

---

## Q1. Defaults in the main ecosystems (2025–2026)

### Takeaway
All three ecosystems default to size-factor scaling followed by a log with pseudo-count 1:
- Scanpy: `normalize_total` with `target_sum=None`, which scales to the median depth, then natural-log `log1p`.
- Seurat v5: `LogNormalize` with `scale.factor = 10000`, natural-log log1p.
- Bioconductor: `scuttle::logNormCounts` with size factors centered at 1, `log2`, pseudo-count 1, usually with scran pooled size factors.

The main alternatives are Pearson residuals. In Scanpy they are analytic and live under `sc.experimental`. In Seurat they come from `SCTransform`, and v2 is the default flavor of that function in Seurat v5. Every ecosystem then builds the clustering kNN graph on PCA, not on the gene matrix.

### Cited Findings
**Scanpy (docs for v1.12.4)**
- `sc.pp.normalize_total`: "Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization." With `target_sum=None` (the default), each cell ends up with "a total count equal to the median of total counts" before normalization. `target_sum=1e6` gives CPM. `exclude_highly_expressed=False` and `max_fraction=0.05` are the defaults. Inputs can be NumPy or CSR; CSC is not supported. — [scanpy.pp.normalize_total](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html)
- Release 1.12.4 (2026-08-27) fixed a bug: with `target_sum=None`, sparse and dense input computed different `target_sum` values. Release 1.11.2 (2025-05-28) computes the median in memory under Dask. Releases 1.10–1.12.4 announce no default changes to normalize_total, log1p, pca or neighbors. A 1.12.2 note says that "In Scanpy 2.0, all graphs will be calculated slightly differently." — [Scanpy release notes](https://scanpy.readthedocs.io/en/stable/release-notes/index.html)
- `sc.experimental.pp.normalize_pearson_residuals` computes analytic Pearson residuals (Lause et al. 2021) from "a negative binomial offset model with overdispersion theta shared across genes". The default `theta=100`; `theta=np.inf` gives a Poisson model. With the default `clip=None`, residuals are clipped to [−√n_obs, √n_obs]. It expects raw counts and still sits in the experimental namespace in 1.12.4. — [scanpy normalize_pearson_residuals](https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals.html)

**Seurat v5 (docs for v5.6.0)**
- `NormalizeData` defaults to `normalization.method = "LogNormalize"` and `scale.factor = 10000`. LogNormalize means "Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale.factor", then natural-log log1p. Other options are `CLR` (centered log ratio) and `RC` (relative counts, no log; with `scale.factor = 1e6` this is CPM). — [Seurat NormalizeData](https://satijalab.org/seurat/reference/normalizedata)
- SCTransform: "The single command SCTransform() replaces NormalizeData(), ScaleData(), and FindVariableFeatures()." "In Seurat v5, SCT v2 is applied by default" (revert with `vst.flavor = 'v1'`). `SCT$scale.data` "contains the residuals (normalized values), and is used directly as input to PCA". `SCT$counts` holds "corrected" UMI counts and `SCT$data` holds log-normalized corrected counts. It returns 3,000 variable features by default. The vignette uses `dims = 1:30` and says "we often benefit by pushing this parameter even higher." — [Seurat SCTransform vignette](https://satijalab.org/seurat/articles/sctransform_vignette)
- `SCTransform()` defaults: `vst.flavor = "v2"`, `variable.features.n = 3000`, `clip.range = c(-sqrt(ncol(umi)/30), sqrt(ncol(umi)/30))`, `do.correct.umi = TRUE`, `do.center = TRUE`, `do.scale = FALSE`, `return.only.var.genes = TRUE`, `ncells = 5000`. Layers: `counts` holds depth-corrected UMIs, `data` holds log1p(corrected counts), `scale.data` holds Pearson residuals, centered by default. — [Seurat SCTransform reference](https://satijalab.org/seurat/reference/sctransform)
- In standalone `sctransform::vst()`, `res_clip_range` defaults to `c(-sqrt(ncol(umi)), sqrt(ncol(umi)))` and `residual_type` to `'pearson'`. `vst.flavor = NULL` means the original model, so v2 is a Seurat-wrapper default. — [sctransform vst docs](https://rdrr.io/cran/sctransform/man/vst.html)

**Bioconductor / OSCA**
- `scuttle::logNormCounts` defaults: `log = TRUE` ("log2-transformed"), `pseudo.count = 1`, and `center.size.factors = TRUE` ("size factors should be centered at unity before being used"). The output assay is `logcounts`. — [scuttle logNormCounts](https://rdrr.io/bioc/scuttle/man/logNormCounts.html)
- The OSCA basic normalization chapter presents library-size factors as "the simplest strategy" and uses `quickCluster()` + `scran::computeSumFactors()` (pooling/deconvolution) in its worked code. The resulting log-values "will be the basis of our downstream analyses". It also says: "The log-transformation is useful as differences in the log-values represent log-fold changes in expression." — [OSCA basic: Normalization](https://bioconductor.org/books/release/OSCA.basic/normalization.html)
- OSCA advanced: "Common practice is to use a pseudo-count of 1, for the simple pragmatic reason that it preserves sparsity." Centering size factors at 1 makes the pseudo-count "an extra read or UMI for each gene". A constant pseudo-count on CPM-scaled values never gets less influential with deeper sequencing. The chapter concludes the log transform is a good default for simplicity and interpretability. It is skeptical of sctransform-style removal of the mean–variance trend, which may also remove log-fold changes of interest. — [OSCA advanced: Normalization, redux](https://bioconductor.org/books/3.14/OSCA.advanced/more-norm.html) (3.14 edition of the book)
- `transformGamPoi` (Ahlmann-Eltze) provides `acosh_transform()`, `shifted_log_transform()` and `residual_transform()` (Pearson / randomized quantile). It names scuttle's `logNormCounts()` as an alternative to `shifted_log_transform()`. — [transformGamPoi GitHub](https://github.com/const-ae/transformGamPoi)

### Inferences
- Effective pseudo-counts differ by ecosystem. Scanpy's median target gives an effective pseudo-count of about 1 at the median cell. Seurat's 1e4 scale factor corresponds to y0 ≈ 0.5 for ~5k-UMI cells (see Q2). scuttle uses log2 with pseudo-count 1 at a mean size factor of 1. These differences rescale the output but do not change its sparsity or sign.
- Standard practice in Seurat v5 is split: `NormalizeData` is the function default, `SCTransform` (v2) is the alternative workflow, and it uses v2 by default.

### Gaps
- I did not re-fetch Seurat `ScaleData` defaults (my recollection is center = scale = TRUE, `scale.max = 10`) or the current Scanpy clustering tutorial. These are not verified this session.
- Scanpy 2.0 graph changes are only announced in the release notes. I found no details.

---

## Q2. Ahlmann-Eltze & Huber 2023 (Nat Methods 20:665–672) and whether 2024–2026 benchmarks confirm or overturn it

### Takeaway
Ahlmann-Eltze & Huber compared 22 transformations from four families. The simple shifted logarithm log(y/s + y0) followed by PCA did as well as or better than residual-based, latent-expression and count-factor-model methods. It also kept the data sparse. No high-profile 2024–2026 benchmark overturns this. A few lower-profile studies favor other methods in specific settings (Dino for 10x clustering; the PFlog/shifted-CLR preprint), but they use different pipelines and metrics, and one is not peer-reviewed.

### Cited Findings
**Ahlmann-Eltze & Huber 2023**
- They compare 22 transformations in four families, plus negative controls of raw counts y and y/s:
  - delta-method: acosh, shifted log variants, CPM-log;
  - residuals: clipped and unclipped Pearson, analytic Pearson, randomized quantile;
  - latent expression: Sanity Distance, Sanity MAP, Dino, Normalisr;
  - count factor analysis: GLM-PCA, NewWave.
  — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Headline: "the shifted logarithm transformation log(y/s+y0) with y0 = 1 followed by PCA, performed surprisingly well". Latent-expression methods and count factor models "performed worse than the shifted logarithm with a reasonable range of PCA dimensions". The abstract says the log with pseudo-count + PCA "performs as well or better than the more sophisticated alternatives". — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138); [Nature Methods](https://www.nature.com/articles/s41592-023-01814-1)
- Pseudo-count ↔ overdispersion: they suggest y0 = 1/(4α). CPM (L = 10⁶) at ~5,000 UMI/cell corresponds to y0 = 0.005, i.e. α = 50, "two orders of magnitude larger than the overdispersions seen in typical single-cell datasets". Seurat's L = 10⁴ corresponds to y0 ≈ 0.5 and α ≈ 0.5. They recommend against CPM input to the shifted log. — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Depth confounding persists after scaling. Their test data were droplets holding a homogeneous RNA solution, so only technical variation exists. Even there, "despite the size factor scaling, after the delta method-based transformation, the size factor remained a strong variance component in the data." — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- They "advise against scaling the results of the shifted logarithm by the sum of the transformed values per cell". It "failed to remove the confounding effect of the sequencing depth" and "did not improve the k-NN recall". — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- There is "no evidence that additional post-processing steps (rescaling the output of the shifted logarithm, selecting HVGs or equalizing the variance of all genes using z scoring) improved the results." — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Sparsity: "in terms of memory consumption, the delta method-based transformations were most attractive because they retained the sparsity of the data". transformGamPoi maps y = 0 to 0. — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- The negative controls (y, y/s), "for which we expect poor performance", were poor. Examples: 34 of 50 neighbors for the best method vs 9 of 50 for a negative control on SUM149PT, and 2.9 vs 1.5 of 50 on HEK downsampling. A log(y+1) variant without size factors was not benchmarked. — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Benchmark mechanics: transform → PCA (4–8 dimension settings) → "the k-NNs of each cell as measured by Euclidean distance", with k = 10, 50, 100. Overlap is scored against simulation ground truth, a split-data reference and deep-sequenced references. — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Analytic Pearson residuals performed like exact Pearson residuals and ran almost as fast as the shifted log. — [Nature Methods](https://www.nature.com/articles/s41592-023-01814-1) (search-snippet level)
- The Huber group's 2025 Nature Genetics LEMUR paper preprocessed data with `transformGamPoi::shifted_log_transform` defaults, so the authors still use this method. — [lemur2025.pdf](https://www.huber.embl.de/pub/pdf/lemur2025.pdf) (search-snippet level)

**Earlier context (2019–2021)**
- Lause, Berens & Kobak 2021 (Genome Biology) argue that Hafemeister & Satija's claim that residuals reduce depth variance in the embedding was "mostly due to the large factor" used in log scaling. They note "large scale factors and/or small pseudocounts … are known to introduce spurious variation". In PBMC t-SNEs, residuals and log showed "similar amount of sequencing depth variation". — [PMC8419999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419999/)
- Hafemeister & Satija 2019: "the low-dimensional representation of log-normalized data was confounded by cellular sequencing depth". They used log(10,000·UMI_gene/UMI_cell + 1). — [PMC6927181](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927181/)

**2024–2026 studies**
- Ge et al., PLOS ONE, 18 Dec 2025. Six methods: Seurat LogNormalize, LIGER, Dino, SCnorm, scTransform, scran. Pipeline: Seurat v4.3, 2,000 HVGs, 50 PCs, Louvain; scored by ARI/AMI etc. on 7 real and 4 simulated datasets. Findings: "Dino Normalization method demonstrated superior performance in clustering analyses of 10× datasets"; scTransform was best for SMART-seq2; SCnorm suits small data. It does not cite Ahlmann-Eltze & Huber. — [PLOS ONE e0335102](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335102)
- This conflicts in part with Ahlmann-Eltze & Huber, who found Dino and the other latent-expression methods below the shifted log. The designs differ: label-based ARI vs kNN overlap. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335102) vs [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Booeshaghi, Hallgrímsdóttir, Gálvez-Merchán & Pachter, preprint "Depth normalization for single-cell genomics count data", later retitled "Normalization for sampled count data" (v4).
  - It proposes PFlog / PFlog1pPF: proportional fitting (PF), then log1p, then a second per-cell PF. The authors call it equivalent to a shifted centered log-ratio and "the only feature-relabeling-equivariant method satisfying the three desiderata": variance stabilization, depth normalization and within-cell monotonicity.
  - They report beating the shifted log on the Ahlmann-Eltze & Huber kNN benchmark. Snippets give a mean kNN Jaccard of 0.677 for log1pPF vs 0.756 for PFlog1pPF. They also criticize sctransform for scrambling within-cell gene rank order.
  - Status: not peer-reviewed as far as I found; numbers are unverified.
  — [bioRxiv v2](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v2); [bioRxiv v4](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4.full)
- A 2026 Computational Biology and Chemistry article (ScienceDirect pii S1476927126002252) tested 465 pipelines (5 normalizations × 4 clustering algorithms) on 3 cancer datasets with about 434k cells. It reports that log-normalized pipelines had low "Biological Discordance Score" (< 0.3) while SCTransform pipelines had high discordance (> 0.6) "despite competitive Borda scores". — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1476927126002252) (snippet only; fetch returned 403)
- A 2025 SciTePress conference paper reports that simple methods (log2, Freeman–Tukey, scran) beat scRNA-specific ones for dimensionality reduction and clustering on small and medium datasets. — [SciTePress 2025](https://www.scitepress.org/Papers/2025/133187/133187.pdf) (snippet only)
- An AusDM'25 paper (arXiv 2601.00277, Jan 2026) found that the best normalization depends on the integration method used and gave no specific normalization recommendation. — [arXiv 2601.00277](https://arxiv.org/abs/2601.00277)

### Inferences
- Taken together, the 2023 result stands: the shifted log with a sensible pseudo-count and size factors is the default to beat. The challengers are lower-profile (PLOS ONE, a conference paper) or not peer-reviewed (PFlog). None reproduces the Ahlmann-Eltze & Huber kNN-overlap design independently except the PFlog preprint, which is by the method's own authors.
- The Ahlmann-Eltze & Huber metric (Euclidean kNN overlap after PCA) is close to what CosTaL+ needs, since CosTaL+ is a kNN-graph method. Their conclusions transfer better than label-ARI benchmarks do, but they were scored on PCA output, not on gene-level cosine.

### Gaps
- I could not retrieve the full PFlog v4 text (bioRxiv returned HTTP 429 repeatedly). Unresolved: the exact definition of the final PF step (multiplicative vs additive per cell), the sign and sparsity of its output, and the exact benchmark numbers. A search summary claimed ~36.8/50 vs ~5.8/50 neighbors on downsampling; that looks inconsistent with the ~2.9/50 scale reported by Ahlmann-Eltze & Huber, so treat it as unverified.
- I found no 2024–2026 Nature Methods or Genome Biology benchmark dedicated to transformations.

---

## Q3. What the scverse best-practices guide recommends (Heumos et al. 2023, Nat Rev Genet 24:550–572, and the sc-best-practices book)

### Takeaway
The guide is task-dependent:
- shifted logarithm for variance stabilization before dimensionality reduction (and DE);
- scran pooling for batch-correction tasks;
- analytic Pearson residuals for HVG selection and rare cell types.

It explicitly warns against CPM input to the shifted log, and it says to cluster with Leiden on a kNN graph built from a low-dimensional representation.

### Cited Findings
- Heumos et al.: "The shifted logarithm transformation log(y/s+1) with size factor s performs well" but "should not be used with counts per million as an input, as it reflects an unrealistically large overdispersion". Also: "Scran performs well for batch correction tasks". Analytical Pearson residuals are "better suited for selection of biologically variable genes". "the normalization method should be chosen carefully and based on the subsequent analysis task." — [Heumos et al. 2023, PMC10066026](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066026)
- Heumos et al.: clustering methods "are applied to the KNN graph computed on a low-dimensional representation of the data". "We recommend using the Leiden algorithm at different resolutions". — [PMC10066026](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066026)
- Book code for the shifted log is `sc.pp.normalize_total(adata, target_sum=None, inplace=False)` then `sc.pp.log1p`, i.e. a median-depth target. It shows scran via R `computeSumFactors(..., clusters=..., min.mean=0.1)` + log1p, and Pearson residuals via `sc.experimental.pp.normalize_pearson_residuals`. — [sc-best-practices: Normalization](https://www.sc-best-practices.org/preprocessing-visualization/normalization/)
- Book text: shifted log is good for "stabilizing variance for subsequent dimensionality reduction and identification of differentially expressed genes". Fixed targets such as 10⁵–10⁶ "can lead to much larger overdispersions than typically seen in single-cell datasets". Pearson residuals can be positive or negative and "do not require downstream heuristic steps like pseudo-count addition or log transformation". "always evaluate based on the subsequent analysis task". — [sc-best-practices: Normalization](https://www.sc-best-practices.org/preprocessing-visualization/normalization/)

### Inferences
- In practice the guide treats the shifted log as the working default, with Pearson residuals as a specialist tool. Neither the paper text nor the book names a single "default".

### Gaps
- The fetched part of the Heumos text did not mention sctransform; the remaining ~180k characters were not read.

---

## Q4. Is log1p without size-factor normalization (original CosTaL) acceptable today, and what problems does it cause?

### Takeaway
No. Every current default (Scanpy, Seurat, OSCA, sc-best-practices, Heumos 2023) applies size-factor scaling before the log. I found no 2023–2026 source that endorses log1p on raw counts for clustering. Even with size factors, depth stays a strong variance component and can drive PC1. Without them, depth and detection rate feed directly into both the direction and the norm of each cell vector, so cosine is not depth-invariant and Tanimoto amplifies the depth signal.

### Cited Findings
- Ahlmann-Eltze & Huber: even after size-factor scaling, "the size factor remained a strong variance component" on technical-only data. The unnormalized negative controls (raw y) performed poorly. — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Hicks, Townes, Teng & Irizarry (Biostatistics 2018; older):
  - "detection rates correlate with the first two principal components" (citing Finak et al.), confirmed across their 15 datasets;
  - "the first PC strongly correlates with the detection rate" (Patel et al. data);
  - under log(x + c) the mean "will depend on the detection rate";
  - correlation remains strong even after centering;
  - "distortion in distance calculations, which in turn can lead to false discoveries when using unsupervised clustering";
  - cells from one tumor split into two clusters by batch because of detection-rate differences.
  — [bioRxiv 025528v4](https://www.biorxiv.org/content/10.1101/025528v4.full)
- Townes et al. 2019 (Genome Biology; older): normalization and transformation can make the per-cell fraction of zeros the largest source of variation, which can produce false-positive clusters (Fig. 3). — [Genome Biology Fig. 3](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1861-6/figures/3) (snippet-level)
- An IA-SVA Bioconductor vignette notes that PC1 of log-transformed counts is often highly correlated with geometric library size (r ≈ 0.9). — [iasva vignette](https://bioconductor.org/packages/release/bioc/vignettes/iasva/inst/doc/detecting_hidden_heterogeneity_iasvaV0.95.html) (snippet-level; older)
- Hafemeister & Satija 2019: even log-normalized data (scale 1e4) gave embeddings where "cells within a cluster are ordered along a gradient that is correlated with sequencing depth". — [PMC6927181](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927181/)
- OSCA: even after scaling, log-transformation can create "trajectories correlated strongly with library size" (Lun 2018). The remedy is downsampling high-coverage cells, which is statistically inefficient, so OSCA recommends it only when such trajectories appear. — [OSCA advanced](https://bioconductor.org/books/3.14/OSCA.advanced/more-norm.html)
- Lause et al. 2021: "sequencing depth normalization and square-root or log-transformation do not affect the zeros, preserving the sparsity of the matrix". Adding size factors therefore costs CosTaL nothing in sparsity. — [PMC8419999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419999/)
- One part of the original CosTaL pipeline is fine: `flavor='seurat_v3'` HVG selection on raw counts. Scanpy documents that this flavor expects count data. — [scanpy highly_variable_genes](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html) (docstring recalled; not re-fetched this session)

### Inferences (mechanism; my derivations)
- **The row norm of log1p(raw) grows monotonically with depth.** Scale a cell's depth by k: detected entries grow by about log k, and more genes become detected, i.e. more non-zeros. So ‖x‖ rises with depth on two counts. In droplet data this is mostly technical (capture and sequencing), plus some real RNA-content biology such as cell size and cycling. The two cannot be separated from counts alone.
- **Cosine is not depth-invariant here.** log1p is not homogeneous: log1p(k·y) ≠ k·log1p(y). For well-detected genes, log1p(k·y) ≈ log k + log y, which adds a constant offset on every detected gene and tilts the vector toward the 0/1 detection pattern. Deeper cells also detect more genes. So depth leaks into the direction of the vector too, which is the Hicks et al. detection-rate effect.
- **Tanimoto penalizes cross-depth edges.** T = cos/(r + 1/r − cos), so a deep and a shallow cell of the same type get down-weighted edges. That can split clusters along depth or detection rate, or make shallow cells peripheral. Here the "extra information" Tanimoto adds over cosine is largely technical.
- **Adding size factors is a one-line change that keeps sparsity.** Use `normalize_total(target_sum=None)` before `log1p`, or scran pooling. It turns the row norm from mostly depth into a mix of profile concentration and residual depth (see Q6).

### Gaps
- I found no current paper that benchmarks log1p(raw) without size factors specifically. Ahlmann-Eltze & Huber benchmarked only y and y/s as negative controls. The "unacceptable" verdict rests on the absence of this method from every current default plus the mechanism evidence above. It would be worth a direct experiment in CosTaL+: on cosine kNN of log1p(raw) vs log1p(y/s), check the correlation of row norm with total UMI and with n_genes, and compare ARI.

---

## Q5. What matrix clustering kNN graphs are built on, and whether cosine vs Euclidean matters

### Takeaway
Current pipelines build the kNN graph on PCA: Scanpy uses 50 PCs by default; Seurat's `FindNeighbors` defaults to 10 PCs and its vignettes use 30 for SCT. PCA output is dense and signed. The default metric is Euclidean in both Scanpy and Seurat, with cosine optional. CosTaL's gene-level sparse cosine kNN is unusual in current practice. I found no 2023–2026 single-cell benchmark that settles cosine vs Euclidean.

### Cited Findings
- `sc.pp.neighbors`:
  - `n_neighbors=15`;
  - with `use_rep=None`, "For .n_vars < N_PCS (default: 50), .X is used, otherwise the PCA representation", and PCA is computed if it is missing;
  - `metric` "defaults to 'euclidean'"; `cosine`, `correlation` and others are accepted;
  - `method='umap'` is the connectivity kernel (`gauss` and `jaccard` are alternatives);
  - with the default `transformer=None`, "For small data, we will calculate exact kNN, otherwise we use" PyNNDescent (approximate).
  — [scanpy.pp.neighbors](https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pp.neighbors.html)
- Seurat `FindNeighbors` (v5.6.0): `dims = 1:10` on `reduction = "pca"`, `k.param = 20`, `nn.method = "annoy"` (approximate), `annoy.metric = "euclidean"` (options: euclidean, cosine, manhattan, hamming), `l2.norm = FALSE` ("Take L2Norm of the data"), and an SNN built by Jaccard with `prune.SNN = 1/15`. — [Seurat FindNeighbors](https://satijalab.org/seurat/reference/findneighbors)
- The SCTransform vignette runs RunPCA → FindNeighbors/RunUMAP `dims = 1:30` and recommends going higher. — [Seurat SCTransform vignette](https://satijalab.org/seurat/articles/sctransform_vignette)
- Heumos et al. 2023: clustering uses "the KNN graph computed on a low-dimensional representation of the data". — [PMC10066026](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066026)
- Benchmarks also score on PCA:
  - Ahlmann-Eltze & Huber: Euclidean kNN after PCA — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
  - Lause et al.: 50 PCs, kNN classifier with k = 15 — [PMC8419999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419999/)
  - Ge et al. 2025: 50 PCs + Louvain — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335102)
- Tool guidance on metric choice (not benchmarks):
  - The `jvecfor` Bioconductor vignette presents Euclidean as the default, fitting when distances carry meaning. It prefers cosine when only direction matters, "for example log-normalized counts whose library-size effects inflate vector norms". — [jvecfor vignette](https://bioc-release.r-universe.dev/jvecfor/doc/jvecfor_vignette.Rmd) (snippet-level)
  - `bullkpy` implements cosine by L2-normalizing and then running Euclidean. — [bullkpy neighbors](https://bullkpy.readthedocs.io/en/latest/api/tl/neighbors.html) (snippet-level)

### Inferences
- **Cosine and Euclidean are closely linked.** On L2-normalized vectors they give identical neighbor rankings: ‖â − b̂‖² = 2 − 2cos (exact identity). Seurat's `l2.norm = TRUE` plus Euclidean therefore equals cosine kNN.
- **On centered PCA scores the two differ in one way.** Euclidean keeps the radial component (distance from the centroid); cosine discards it. With 30–50 PCs and moderate k I would expect small differences in clustering, but I found no current benchmark that quantifies this.
- **Gene-level kNN vs PCA kNN is a bigger design choice than the metric.** PCA denoises: Ahlmann-Eltze & Huber's positive results hold only "with a reasonable range of PCA dimensions". A gene-level cosine kNN on 2,000 HVGs, as in CosTaL, sits outside the setting where the shifted log was validated.

### Gaps
- No 2023–2026 single-cell benchmark of cosine vs Euclidean. Older work, not re-verified this session, includes Kim et al. 2019 (Brief Bioinform, "Impact of similarity metrics on single-cell RNA-seq data clustering"), reportedly favoring correlation-based measures, and Skinnider et al. 2019 (Nat Methods, measures of association).
- The OSCA clustering chapter's default (`clusterCells`/`buildSNNGraph` on PCA, Euclidean) was not re-fetched.

---

## Q6. Output properties for each transformation: (a) sparse? (b) can be negative? (c) row-norm semantics

### Takeaway
Two clean classes:
- **Sparse, non-negative** (exact zeros kept): log-family transforms with pseudo-count 1 or a sparsity-preserving variant: log1p(raw), shifted log, scran + logNormCounts, acosh/transformGamPoi shifted log, Seurat RC/LogNormalize, SCT `data`. These suit CosTaL+'s sparse inverted-index backend. Their row norm is not normalized away.
  - For log1p(raw), the row norm is mostly depth.
  - With size factors, it mixes biology (how concentrated the profile is over the chosen genes) with residual depth and detection effects.
- **Dense, signed** (centered): Pearson residuals (analytic and SCT `scale.data`), z-scored data, PCA scores, latent-expression outputs, GLM-PCA factors. These need a dense backend. Their row norm measures how far a cell sits from the dataset's average profile, which is dataset-relative and only partly depth-free.

### Cited Findings (property-level evidence)
- Log/sqrt after depth scaling "do not affect the zeros, preserving the sparsity". "Pearson residuals form a dense matrix without any zeros and so can take a large amount of memory to store (4.5 Gb for the PBMC dataset)". Selecting 1,000 genes cuts this to about 0.3 Gb, and residual variance can be computed per gene without storing the matrix. — [Lause et al. 2021, PMC8419999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419999/)
- Analytic residual formula: Z_cg = (X_cg − μ̂_cg)/√(μ̂_cg + μ̂_cg²/θ), with μ̂_cg = n_c·p̂_g (row sum × column sum / total) and θ = 100. — [PMC8419999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419999/)
- Clip ranges: ±√n_obs in Scanpy — [scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals.html); ±√(N/30) in Seurat SCTransform — [Seurat](https://satijalab.org/seurat/reference/sctransform); ±√N in sctransform `vst` — [sctransform](https://rdrr.io/cran/sctransform/man/vst.html)
- sctransform Pearson residuals: "gene variance was strikingly consistent across cells with different sequencing depths". For genes without biological heterogeneity, residuals are expected to have "a mean of zero and unit variance". — [Hafemeister & Satija 2019, PMC6927181](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927181/)
- Delta-method transforms keep sparsity, and transformGamPoi maps 0 → 0. The shifted log approximates acosh with y0 = 1/(4α). Sanity and Normalisr output logarithmic latent expression; Sanity MAP returns the posterior maximum, and Dino "returns random samples from the posterior". — [PMC10172138](https://pmc.ncbi.nlm.nih.gov/articles/PMC10172138)
- Seurat `SCTransform` stores `data` as log1p(corrected counts) and `scale.data` as Pearson residuals, centered (`do.center = TRUE`) and not scaled, for 3,000 variable genes by default (`return.only.var.genes = TRUE`). — [Seurat SCTransform reference](https://satijalab.org/seurat/reference/sctransform)
- OSCA uses pseudo-count 1 because "it preserves sparsity". — [OSCA advanced](https://bioconductor.org/books/3.14/OSCA.advanced/more-norm.html)
- Pearson residuals are positive when there are more counts than expected and negative when fewer. — [sc-best-practices](https://www.sc-best-practices.org/preprocessing-visualization/normalization/)

### Property table
(a), (b) and the Basis column come from the sources above. Column (c) is my reasoning from the formulas unless a source is named.

| Transformation (status 2023–26) | (a) Sparse? | (b) Negative? | (c) What the per-cell row norm encodes | Basis |
|---|---|---|---|---|
| Raw counts y (negative control) | Yes | No | Dominated by depth and a few highly expressed genes. Technical. | AE&H controls |
| **log1p(y), no size factors (original CosTaL; not a current default)** | Yes (log1p(0) = 0) | No | Grows monotonically with depth: entries shift by ~log k and more genes are detected. Mostly technical, with real RNA content mixed in. Depth also changes the direction, so cosine is not depth-invariant. | Hicks 2018; AE&H; my derivation |
| y/s scaling only (CPM / Seurat RC / PF; no log) | Yes | No | L1 norm fixed by construction. L2 norm = L·√(Σp̂²), i.e. profile concentration (biological dominance by few genes), plus a low-depth upward bias, since E[Σp̂²] = Σp² + (1 − Σp²)/n_c under multinomial sampling (my derivation). | Seurat NormalizeData; AE&H |
| **Shifted log: log(y/s + 1)** — Scanpy normalize_total(target_sum=None)+log1p; Seurat LogNormalize (L = 1e4, ln); scuttle logNormCounts (log2, PC 1, SF centered); scran pooling + log. **Current default.** | Yes | No | Not normalized: neither L1 nor L2 is fixed after the log. Mixes biology (profile concentration and level over the HVG subset, e.g. cells dominated by a few markers) with residual depth/detection effects, since AE&H find the size factor "remained a strong variance component". How the norm depends on depth can be non-monotonic: one UMI in a shallow cell maps to log1p(L/n_c), which is large when n_c < L. | AE&H; OSCA; Lause |
| Shifted log with CPM (L = 1e6) | Yes | No | Strongly driven by the detection pattern: the effective pseudo-count is tiny, so the zero/non-zero gap dominates. **Discouraged** (AE&H, Heumos). | AE&H; Heumos |
| acosh / transformGamPoi shifted_log (y0 = 1/(4α)) | Yes (0 → 0) | No | Same as the shifted log. | AE&H; transformGamPoi |
| Shifted log ÷ per-cell sum of transformed values | Yes | No | L1 fixed, so the norm is largely normalized away. AE&H **advise against** it: it does not remove depth confounding and does not help kNN recall. | AE&H |
| PFlog1pPF / shifted CLR (Pachter preprint) | Unverified (yes if the final PF is multiplicative; no if it is CLR-style centering) | Unverified (as above) | If PF fixes each cell's sum of log values, the L1 norm is normalized away. **Gap.** | bioRxiv preprint |
| Seurat CLR (`NormalizeData` method; mainly for ADT) | Likely yes (log1p(0/g) = 0) | No | Per-feature or per-cell geometric-mean scaling; not used for RNA clustering. | Seurat docs |
| SCT `data` layer = log1p(corrected counts) | Yes | No | Like the shifted log, but counts are first corrected to a common depth, so less depth signal is left in the norm. | Seurat SCTransform |
| **Analytic Pearson residuals** (Scanpy experimental; θ = 100, clip ±√n) | **No: dense**, "without any zeros" | **Yes**: a zero count gives −μ/√(μ + μ²/θ), whose magnitude grows with depth | Under the null each Z has ~unit variance, so ‖Z‖² ≈ G for a typical cell. Extra norm = deviation from the global average profile in noise units. For a real fold change f, Z ≈ √(n_c p_g)(f − 1) until μ ≳ θ, so the norm grows ~√depth for distinct cells, then is capped by θ and clipping. Biological atypicality weighted by statistical evidence; dataset-relative. | Lause 2021; scanpy docs; my derivation |
| **SCTransform v2 `scale.data`** (Seurat v5 default flavor) | **No: dense** (3,000 genes) | **Yes**: centered, clipped ±√(N/30) | Same meaning as the analytic residuals, with explicit per-gene centering. Measures distance from the dataset mean profile, built to be depth-independent in variance. | Seurat docs; Hafemeister 2019 |
| z-scored shifted log (sc.pp.scale / Seurat ScaleData) | **No: dense** after centering | **Yes** | Diagonal-Mahalanobis distance from the dataset centroid (atypicality), dominated by rare high-variance genes unless clipped. AE&H found no benefit from z-scoring. | AE&H |
| **PCA scores, 30–50 PCs (the actual kNN input in Scanpy/Seurat/OSCA)** | **No: dense** | **Yes** (centered) | Distance from the centroid within the top-PC subspace. Biology plus whatever technical axis PCA keeps: if PC1 tracks depth or detection (Hicks 2018; AE&H), the norm carries depth. Cosine drops the radial part; Euclidean keeps it. | Scanpy/Seurat docs; Hicks 2018 |
| Latent expression: Sanity (MAP/Distance), Dino, Normalisr | Likely dense: posterior estimates and samples fill the zeros (inference) | Sanity/Normalisr are on a log scale, so values are likely negative or mixed (inference). Dino samples are non-negative on the count scale before any log (inference). | Depends on the method. AE&H found these worse than the shifted log + PCA. | AE&H; **Gap** |
| GLM-PCA / NewWave factors | Dense | Yes | Factor scores (like PCA); centering semantics depend on the model. | AE&H |

### Inferences for CosTaL+ design
1. **Backend choice follows the input class.** Use the sparse inverted-index exact cosine backend only for the sparse, non-negative class: log1p(raw), shifted log, scran/logNormCounts, acosh, SCT `data`. Use a dense exact backend (blocked GEMM) for PCA scores (n × 30–50), Pearson residuals, SCT `scale.data` and z-scored data. Dense Pearson residuals on 2,000 genes cost n × 2,000 × 4 bytes in float32, e.g. ~8 GB for 1M cells (my arithmetic). Lause recommends restricting to HVGs.
2. **Whether Tanimoto adds anything depends on the transformation.**
   - log1p(raw): the extra signal is mostly depth and detection, i.e. technical. Expect it to hurt by penalizing same-type pairs that differ in depth.
   - Shifted log (sparse): mixed. Profile concentration and marker dominance are biological; residual depth is technical. Tanimoto can plausibly help, but this should be measured.
   - PCA or residuals (dense, centered): the norm is distance from the dataset centroid. Tanimoto then penalizes pairs at different "radii". cos can be negative, so T ∈ [−1/3, 1], and edge weights need clipping or a convention for T < 0.
3. **Upgrade the default scRNA preprocessing.** The minimal fix that keeps everything sparse and non-negative is to insert `normalize_total(target_sum=None)` (or scran size factors) before `log1p`. Keep `seurat_v3` HVG on raw counts. Never use CPM. A mainstream-compatible alternative is shifted log → PCA (50) → dense exact kNN.
4. **Suggested validation experiment** (not found in the literature): for each preprocessing option, correlate the row norm with log total UMI and with n_genes, and with cell-type labels (e.g. η² of the norm by label). Then compare ARI for cosine-only vs Tanimoto-reweighted graphs. This directly measures whether the norm is "informative" in the sense CosTaL+ needs.

### Gaps
- Sanity, Dino and Normalisr output sparsity and sign were not confirmed from their own documentation; the table entries are marked as inference.
- I found no empirical study that quantifies how much row-norm variance in shifted-log HVG matrices is biological vs technical.
- The PFlog1pPF output properties are unresolved (bioRxiv returned HTTP 429 every time).
- In the PMC extractions of Hafemeister 2019 and Lause 2021, the residual clip appeared as "±N" or "±n"; the square root was probably lost in extraction. Current code and docs confirm ±√n (Scanpy, sctransform `vst`) and ±√(N/30) (Seurat SCTransform).
