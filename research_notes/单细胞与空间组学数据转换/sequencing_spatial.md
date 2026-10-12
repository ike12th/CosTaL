# Count transformations / normalization for sequencing-based spatial transcriptomics (Visium, Visium HD, Stereo-seq, Slide-seq / Curio Seeker), 2023–2026 practice

Scope note: researched October 2026. "Verified" = read on the cited page (docs/paper/PMC full text) in this session; "snippet" = only seen via search-result summaries, not read in full. Items under **Inferences** are my own reasoning (including small arithmetic done in this session), not claims from sources. CosTaL+ context: exact cosine kNN, then Tanimoto edge weight T = a·b / (‖a‖² + ‖b‖² − a·b); sparse vs dense kNN backend.

## Q1. What do the standard pipelines / tutorials use by default (Space Ranger, Scanpy/Squidpy, Seurat, Giotto, BANKSY, Stereopy, OSTA/Bioconductor)?

### Takeaway
Every vendor pipeline hands over **raw integer UMI count matrices** (Space Ranger bins/cells, Stereo-seq GEF, Slide-seq/Curio beads), and essentially every mainstream tutorial in 2024–2026 applies **global library-size scaling + log1p** (Seurat `NormalizeData`/LogNormalize, Scanpy `normalize_total`+`log1p`, Stereopy `normalize_total(1e4)`+`log1p`, Giotto scalefactor 6000 + log2) — the one prominent exception is Seurat's *older* Visium/Slide-seq vignette, which uses **SCTransform**, and BANKSY, which uses **library-size-scaled counts without log** followed by internal z-scaling/PCA. Seurat's 2024–2025 Visium HD vignettes (bins and cell segmentations) switched to plain LogNormalize.

### Cited Findings
**Seurat (Satija lab)**
- Visium HD vignette (bins): loads 8 µm and 16 µm bins by default — `Load10X_Spatial(data.dir = localdir, bin.size = c(8, 16))`; "10x recommends the use of 8μm binned data for analysis"; normalization is `NormalizeData(object)` with the stated choice "we use standard log-normalization for spatial data" (not SCTransform). 2 µm is described as the native size: "since the data from this resolution is sparse, adjacent bins are pooled together"; also "many spots have very few counts, in-part due to low cellular density in certain tissue regions". Downstream it uses sketching (`SketchData`, a subsampling step, not a normalization), BANKSY and RCTD on 8 µm. — [Seurat Visium HD vignette](https://satijalab.org/seurat/articles/visiumhd_analysis_vignette)
- Visium HD **cell-segmentation** vignette (requires Seurat ≥ 5.4 / SeuratObject 5.3, released Dec 2025): `Load10X_Spatial(data.dir = path_to_data, bin.size = c(8, "polygons"), ...)`, default assay `Spatial.Polygons`, then "normalize the data with standard log-normalization" via `NormalizeData(kidney_obj)`. Kidney example: 666,695 8 µm bins vs 148,056 segmented cells; example per-cell nCount values ~107–814. — [Seurat cell-segmentation vignette](https://satijalab.org/seurat/articles/visiumhd_analysis_cell_segmentations); version/date info — [search summary of Seurat Visium HD pages](https://satijalab.org/seurat/articles/visiumhd_commands_intro) (snippet)
- Older (Visium v1 / Slide-seq era, still online) spatial vignette uses **SCTransform** for both: `SCTransform(brain, assay = "Spatial")` and `SCTransform(slide.seq, assay = "Spatial", ncells = 3000)`. Rationale quoted on page: "the variance in molecular counts / spot can be substantial for spatial datasets", and this variance "is also dependent on the tissue anatomy"; sctransform "builds regularized negative binomial models of gene expression in order to account for technical artifacts". For Slide-seq: "many beads contain particularly low UMI counts, but choose to keep all detected beads". — [Seurat spatial vignette](https://satijalab.org/seurat/articles/spatial_vignette)
- SCTransform output layout: new assay with "counts being (corrected) counts, data being log1p(counts), scale.data being pearson residuals". — [Seurat SCTransform reference](https://search.r-project.org/CRAN/refmans/Seurat/html/SCTransform.html) (snippet)

**Scanpy / Squidpy**
- Scanpy spatial tutorial (now marked legacy, "For up-to-date analysis tutorials, kindly check out SquidPy tutorials"): `sc.pp.normalize_total(adata, inplace=True)`, `sc.pp.log1p(adata)`, `sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)`; notes "there are alternatives for normalization" e.g. "SCTransform or GLM-PCA". — [Scanpy spatial basic analysis (1.10.x)](https://scanpy.readthedocs.io/en/1.10.x/tutorials/spatial/basic-analysis.html)
- Squidpy Visium H&E tutorial ships a "pre-processed dataset, with pre-annotated clusters" and says "The pre-processing pipeline is the same as the one shown in the original Scanpy tutorial" (i.e., normalize_total + log1p); no normalization code shown. — [Squidpy Visium H&E tutorial](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_hne.html)

**Giotto**
- `normalizeGiotto` defaults: `norm_methods = c("standard", "pearson_resid", "osmFISH", "quantile")` (first = default), `scalefactor = 6000`, `log_norm = TRUE`, `log_offset = 1`, `logbase = 2`, `scale_feats = TRUE`, `scale_cells = TRUE`, `scale_order = c("first_feats", "first_cells")`. "standard" = library-size normalization with scale factor → log → z-score by genes and/or cells; "pearson_resid" = Lause/Kobak analytic Pearson residuals; results stored in `normalized` and `scaled` slots. — [Giotto normalizeGiotto reference](https://giottosuite.com/reference/normalizeGiotto.html)

**BANKSY**
- Quick start: `computeLibraryFactors(se)` then `normalizeCounts(se, log = FALSE)` into a `normcounts` assay, with the code comment "# Normalization to mean library size" — i.e., scaled but **not log-transformed**; `runBanksyPCA(se, assay_name = "normcounts", ...)` then computes "scaling parameters" for own and neighbourhood expression before PCA. — [Banksy GitHub README](https://github.com/prabhakarlab/Banksy)
- OSTA Visium HD (binned) workflow runs BANKSY (k_geom = 8, lambda = 0.2, 20 PCs) on 3,000 HVGs after `normalizeRnaCounts.se()` (OSTA describes this helper as "log-library size normalization" in the companion chapter). — [OSTA Visium HD binned](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-bin.html); [OSTA Visium HD segmented](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-seg.html)

**Stereopy (Stereo-seq / STOmics)**
- Square-bin quick start: `st.io.read_gef(file_path=..., bin_size=50)`; QC; recommends `data.tl.raw_checkpoint()` ("We strongly suggest to use self.raw to record the raw gene expression matrix"); then `data.tl.normalize_total(target_sum=10000)` + `data.tl.log1p()`; HVG n_top_genes=2000; `data.tl.scale(max_value=10, zero_center=True)`; PCA 30 PCs; Leiden ("we strongly recommend using leiden"). Also offers scTransform and quantile normalization; with `zero_center=False` "sparse matrix will be used for calculation". — [Stereopy Square Bin quick start](https://stereopy.readthedocs.io/en/latest/Tutorials/SquareBin_Clustering.html)
- `normalize_total` default target_sum 10000; `None` → median of totals. — [Stereopy normalize_total API](https://stereopy.readthedocs.io/en/latest/content/stereo.core.StPipeline.normalize_total.html) (snippet)
- Cell-bin tutorial (cell-segmented GEF) uses the same cal_qc → normalize_total → log1p workflow. — [Stereopy Cell Bin quick start](https://stereopy.readthedocs.io/en/latest/Tutorials/CellBin_Clustering.html) (snippet)

**Space Ranger output**
- Visium HD 8 µm and 16 µm bins are gene-wise **sums of UMI counts** of the 2 µm squares; Space Ranger outputs `binned_outputs` at 2, 8, 16 µm and (Space Ranger v4, June 2025) H&E-segmented cell-level `segmented_outputs`. — [OSTA Visium HD binned](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-bin.html) (snippet); [OSTA Visium HD segmented](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-seg.html)

**Curio Seeker / Slide-seq**
- Curio Seeker is "based on Slide-seqV2", using a monolayer of tightly packed **10 µm beads**; vendor HTML report includes clustering, UMAP and spatial cluster maps. — [BioChain Curio Seeker page](https://biochain.com/curio-seeker-whole-transcriptome-spatial-mapping) (snippet; reseller page)

### Inferences
- The de facto community default for sequencing-based spatial in 2024–2026 is "LogNormalize-type" (global size factor → log1p). SCTransform remains common in Seurat users' Visium v1/v2 and Slide-seq work but Seurat's own HD vignettes do not use it.
- Because all vendor outputs are raw counts, CosTaL+ users will usually receive either (i) raw counts, (ii) log1p(CPx) (sparse), or (iii) a dense derived matrix (PCA, Pearson residuals, z-scores, BANKSY PCs, deep-learning embeddings). These are the realistic input classes to design for.

### Gaps
- Could not read the 10x Space Ranger "clustering and dimensionality reduction" algorithm page (HTTP 429 on three attempts), so the exact normalization used by Space Ranger's own built-in PCA/graph-clustering (and whether it runs on 8 µm bins or cells for HD) is unverified here.
- Curio Seeker pipeline's internal normalization (Seurat LogNormalize vs SCTransform) not found in any public doc.
- Did not verify the current Squidpy Visium HD / SpatialData tutorial code.

## Q2. Library size confounded with biology: Bhuva et al. (Genome Biology 2024), SpaNorm, and 2025–2026 follow-ups/benchmarks

### Takeaway
Bhuva et al. (2024) showed that per-spot/per-cell library size is spatially structured and **tracks tissue regions even after accounting for cell number**, so scRNA-seq-style library-size normalization removes domain signal (sctransform worst; "no normalization" = log raw counts, and scran, best for domain detection). SpaNorm (Genome Biology 2025) models library size and biology as separate smooth spatial functions in a per-gene NB GLM and returns library-size-adjusted data. A large 2025 clustering benchmark (iMeta) nonetheless found normalization "indispensable" — the field has **not** converged.

### Cited Findings
**Bhuva et al. 2024 (Genome Biology 25:99, published 18 Apr 2024; Bhuva DD, Tan CW, …, Phipson B, Davis MJ)**
- 25 samples, 4 platforms: 10x Visium (human DLPFC), 10x Xenium (mouse brain, breast cancer), BGI STOmics/Stereo-seq (mouse brain), NanoString CosMx (NSCLC); 18,647 parameter combinations. — [Bhuva et al. 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/)
- Library size varies with structure: cortical layers, white matter, hippocampus have distinct patterns; tumour regions in NSCLC and breast intraductal carcinoma have higher detections per cell. In a Poisson model, **cell number explained most library-size variance, then tissue region; region stayed significant after accounting for cell number (p < 2×10⁻³⁰⁸)**. Authors attribute part of this to tissue-architecture-dependent reagent permeability, hard to separate from true biology. — [Bhuva et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/)
- Compared: no normalization (log-transformed raw counts), scran, sctransform, RUVIII-NB; clustering by SNN graph (walktrap/Louvain/Leiden), BayesSpace, SpaGCN; scored by ARI. **sctransform generally worst** ("removed biological signal along with library size effects"); no-normalization and scran best with BayesSpace and graph clustering; SpaGCN mostly favoured scran; spatially aware methods best for 21/25 samples; scran best normalization for 15/25; RUVIII-NB best for 5/9 cancer datasets. — [Bhuva et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/)
- Quotes: "library size is associated with tissue structure"; normalizing it out with scRNA-seq methods "will negatively affect spatial domain identification"; "Spatial data should not be specifically corrected for library size prior to analysis". Library-size normalization may be reasonable for cross-sample comparison/integration. — [Bhuva et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/)

**SpaNorm (Genome Biology 2025; bioRxiv 2024.05.31.596908; Bioconductor package `SpaNorm`)**
- Per-gene negative binomial model that "decomposes spatially-smooth variation" into a part unrelated to library size (biology) and a part related to library size, using thin-plate splines and GLM-based decomposition; keeps the biological part. Across **27 tissue samples, 6 datasets, 4 platforms** it "outperforms commonly used single-cell normalisation approaches while retaining spatial domain information and detecting spatially variable genes". — [SpaNorm, Genome Biology 2025](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03565-y) (snippet); [SpaNorm GitHub](https://github.com/bhuvad/SpaNorm); [SpaNorm bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.31.596908.full.pdf) (snippet; full text not retrievable, HTTP 429)
- Motivation in preprint: scRNA-seq methods "use global scaling factors and may suffer when the data are confounded by spatial region-specific library size biases". — [SpaNorm bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.31.596908.full.pdf) (snippet)
- Outputs (`adj.method`): **"logpac"** = log percentile-adjusted counts (PAC), stored in the `logcounts` assay; **"pearson"** = "Pearson residuals from factoring out unwanted effects"; **"meanbio"/"medbio"** = mean/median of each gene at each location from the biological component of the fit. Defaults: `df.tps = 6` (spline complexity), `sample.p = 0.25` (fit on 25% of spots/cells), `scale.factor = 1` (vignette example uses 4 to "increase the signal"). Example data: 10x Visium DLPFC (~4,000 spots). — [SpaNorm vignette source (Rmd)](https://bioc-release.r-universe.dev/SpaNorm/doc/SpaNorm.Rmd); [SpaNorm Bioconductor vignette](https://bioconductor.org/packages/release/bioc/vignettes/SpaNorm/inst/doc/SpaNorm.html)
- Version discrepancy: the vignette describes "logpac" as default, while a devel reference manual's usage signature lists `adj.method = "auto"` first. — [SpaNorm vignette Rmd](https://bioc-release.r-universe.dev/SpaNorm/doc/SpaNorm.Rmd) vs [SpaNorm devel manual](https://free.nchc.org.tw/bioconductor/packages/devel/bioc/manuals/SpaNorm/man/SpaNorm.pdf) (snippet)
- Search-surfaced claim (likely SpaNorm paper, not verified in full text): CosMx and STOmics datasets had the highest proportion of genes with region-specific effects, then Xenium; Visium the lowest. — [SpaNorm, Genome Biology 2025](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03565-y) (snippet, attribution uncertain)

**2025 clustering benchmark (iMeta 2025, Chen et al.; 14 methods, ~600 datasets, 10 technologies incl. Visium, Visium HD, Stereo-seq, Slide-seq)**
- 14 methods: BASS, BANKSY, BayesSpace, CCST, CellCharter, DeepST, GraphST, PRECAST, SEDR, STAGATE, SpaGCN, SpaceFlow, SpatialMGCN, stLearn. "normalization was indispensable"; SPARK-X 3,000 SVGs beat HVGs for most methods; "The impact of log transformation and standardization varied considerably"; default 200 PCs "often suboptimal". Recommended general pipeline: "(1) library‐size normalization, (2) log transformation, (3) selection of 3000 spatially variable genes, (4) no standardization, and (5) … top 20 principal components." Large Stereo-seq/Visium HD slices were split because many methods ran out of memory. — [iMeta 2025 benchmark, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12747554/); [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/imt2.70084)

**Related (imaging-based, out of scope but same theme)**
- Atta et al. (Genome Biology 2024): with skewed targeted gene panels, library-size and scTransform normalization give higher DE error rates, up to ~30% in some simulated panels. — [Atta et al. 2024](https://link.springer.com/article/10.1186/s13059-024-03303-w) (snippet)

**Visium HD-specific**
- Oliveira et al. (Nature Genetics 2025, 10x Visium HD CRC paper) call for "spatially aware methods capable of including information from neighboring bins during unsupervised clustering". — [Oliveira et al. 2025](https://www.nature.com/articles/s41588-025-02193-3) (snippet)
- I found **no** 2025–2026 normalization benchmark specific to Visium HD bins/cells, and SpaNorm's published evaluation does not include Visium HD (per search). — search summary, [SpaNorm GB 2025](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03565-y) (snippet)

### Inferences
- Bhuva vs iMeta conflict is plausibly about **what is evaluated**: Bhuva used BayesSpace/SpaGCN/SNN on DLPFC-type annotated domains where library size itself is a domain marker (white matter vs grey matter); iMeta averaged over 14 mostly deep-learning methods and ~600 datasets with mixed annotation types. Both use manual annotations as ground truth, so both can reward/penalize magnitude-driven signal.
- Seurat's spatial vignette motivates SCTransform by exactly the observation Bhuva formalized (depth "dependent on the tissue anatomy"), but draws the opposite conclusion (remove it). Bhuva found sctransform the worst for domain detection — so SCT on spatial data is now contested practice.
- For CosTaL+: in spatial data, **row magnitude of raw/log-raw counts is partly biological at the domain level** (cell density, region), and partly technical (permeabilization, capture). Tanimoto's magnitude sensitivity therefore could help *domain* detection on unnormalized/log-raw input, but would reintroduce the very confound that library-size normalization removes; this is a defensible design choice only if the user's target is domains, not cell states.

### Gaps
- Could not read SpaNorm full text (Springer/bioRxiv blocked), so exact competitor list, quantitative ARI gains, and whether logPAC preserves zeros are unverified.
- No 2026 normalization benchmark for sequencing-based spatial found; no evidence on SpaNorm at Visium HD 2/8 µm or cell level.

## Q3. Visium HD (2 / 8 / 16 µm, cells) and Stereo-seq (bin50, cellbin): sparsity, bin-to-cell methods, and normalization that follows

### Takeaway
Visium HD is a continuous lawn of 2×2 µm barcoded squares (~11 M); 8 µm bins (4×4 squares) carry ~16× the mean UMIs of 2 µm squares and are 10x's recommended default; 2 µm bins are reported (informally) at ~1–10 UMIs, so >99.9% zeros per row is the expected regime. Getting to cells is done by bin2cell (StarDist nuclei + expansion + destriping), ENACT (several bin-to-cell weighting schemes), or Space Ranger v4's own H&E segmentation (June 2025); in all published workflows the resulting cell×gene sums are then **log-normalized** (Seurat LogNormalize, OSTA log-library-size, Scanpy normalize_total+log1p).

### Cited Findings
**Visium HD geometry and depth**
- Capture area is "2 × 2-µm squares of uniquely barcoded oligonucleotides placed directly next to each other"; downstream analyses default to 8 µm, which "provide a 16-fold increase in the average number of UMI" per bin; nucleus-level matrices built by assigning 2 µm bins inside segmented nuclear polygons. — [Oliveira et al., Nature Genetics 2025](https://www.nature.com/articles/s41588-025-02193-3) (snippet)
- 10x technical note: 8 µm gives ~16-fold more mean UMIs than 2 µm; the best bin size depends on tissue; "the 8 µm bin size is an effective starting point for most researchers". — [10x CG000686 tech note](https://cdn.10xgenomics.com/image/upload/v1722285090/support/documents/CG000686_VisiumHDSpatialGeneExpressionPerformance_TechnicalNote_RevA.pdf) (snippet; PDF not parseable here)
- 2 µm grid for the CRC dataset: 3350 × 3350 = 11,222,500 positions at 2,000 nm/px. — [sainsc Visium HD tutorial](https://sainsc.readthedocs.io/v0.4.0/tutorials/VisiumHD_CRC.html)
- Informal: "2 µm data is very sparse, typically 1–10 UMIs per bin". — [Silva-Oliveira blog, 2026](https://rafael-silva-oliveira.github.io/blog/2026/segmentation-and-annotation/) (practitioner blog, not peer-reviewed). A user reports ~4–8 UMIs per **8 µm** bin in mouse skin wound tissue despite 70–90% saturation — [ResearchGate thread](https://www.researchgate.net/post/Why_do_we_see_low_UMI_recovery_on_10x_Visium_HD_with_mouse_skin_wound_tissue_despite_high_sequencing_saturation) (anecdotal)
- OSTA CRC workflow: 545,913 bins at 8 µm and 137,051 at 16 µm, 18,085 genes; QC removes "8 µm bins with a library size below 100" and 16 µm bins with libsize ≤ 400 (expected ~4× increase); bins over tissue gaps have low counts "likely due to ambient RNA or transcript spillover". Main analysis was done at **16 µm**. — [OSTA Visium HD binned](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-bin.html)
- OSTA segmented (Space Ranger v4 cells): median 24 (mean 28.55, max 185) 2 µm bins per cell; median nuclear fraction 0.30; 8 µm bins map to a median of 2 cells (max 8), 16 µm bins to a median of 4 (max 17); example cells ~300–1,400 UMIs; cell area estimated as 4 µm² × number of 2 µm bins; normalization by `normalizeRnaCounts.se()` ("log-library size normalization"); SingleR or RCTD for annotation. — [OSTA Visium HD segmented](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-seg.html)
- Seurat kidney example: 666,695 8 µm bins vs 148,056 segmented cells; example cells nCount ~107–814. — [Seurat cell-segmentation vignette](https://satijalab.org/seurat/articles/visiumhd_analysis_cell_segmentations)

**bin2cell (Polański et al., Bioinformatics 2024, 40:btae546)**
- Works on 2 µm bins; StarDist nuclei expanded into neighbouring unlabelled bins; "secondary labels" from spatial clusters of expression totals where nuclei are missed. — [bin2cell, Bioinformatics](https://academic.oup.com/bioinformatics/article/40/9/btae546/7754061) (snippet)
- 2 µm bins show per-row/per-column capture disparity ("striped" totals). Destriping: "each bin's count total is divided by a user-specified quantile (by default 0.99) from its corresponding array row", then per column, rescaled to count space by the global per-bin quantile; "The count matrix is then by default rescaled per bin to match the corrected count totals"; "The final object accumulates all counts from 2 μm bins by cell labels." "All 113 cell types have a higher average number of genes detected in b2c" than 8 µm bins. — [bin2cell, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419951/)

**ENACT (Bioinformatics 2025, 41:btaf094; Sanofi)**
- Motivation: 8 µm bins often overlap ≥2 cells and cells overlap several bins. Bin-to-cell assignment strategies: naive, weighted_by_area, weighted_by_transcript, weighted_by_cluster, using StarDist boundaries; cell typing via CellAssign/Sargent/CellTypist. — [ENACT, Bioinformatics](https://academic.oup.com/bioinformatics/article/41/3/btaf094/8063614) (snippet); [ENACT Zenodo supplement](https://zenodo.org/records/14748859) (snippet)

**Other**
- sainsc works on the 2 µm grid with a Gaussian KDE (kernel 4 µm) rather than discrete bins; for 16 µm signatures it uses `normalize_total(target_sum=1e4)` + `log1p`. — [sainsc tutorial](https://sainsc.readthedocs.io/v0.4.0/tutorials/VisiumHD_CRC.html)
- A 2026 arXiv review lists "sparse molecular signals" among major segmentation challenges and notes a lack of gold-standard benchmarks. — [arXiv 2606.09675](https://arxiv.org/abs/2606.09675)
- OSTA notes a comparative study of segmentation methods on Visium HD "has yet to be conducted". — [OSTA segmented workflow](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-seg.html) (snippet)

**Stereo-seq**
- Stereopy `bin_size` combines nanoball spots into square bins; bin20 ≈ "approximately 10 or 14 microns" depending on chip center-to-center distance; tutorial reads bin50; normalize_total(1e4) + log1p; cellbin data follow the same workflow. — [Stereopy Square Bin](https://stereopy.readthedocs.io/en/latest/Tutorials/SquareBin_Clustering.html); [Stereopy Cell Bin](https://stereopy.readthedocs.io/en/latest/Tutorials/CellBin_Clustering.html) (snippet)
- In an independent 11-method benchmark (You et al., Nature Methods 2024), Stereo-seq showed high capture efficiency at raw depth; Slide-seq V2 and DynaSpatial after depth normalization; Curio Seeker was included; no method reached read saturation. — [You et al. 2024, Nat Methods](https://www.nature.com/articles/s41592-024-02325-3) (snippet)

### Inferences
- **Arithmetic bound (mine):** a row with U UMIs has at most U non-zero genes, so zero fraction ≥ 1 − U/G. With G ≈ 18,085 (OSTA CRC): U = 5 → ≥ 99.97%; U = 30 → ≥ 99.83%; U = 100 (OSTA 8 µm QC floor) → ≥ 99.45%; U = 300 → ≥ 98.3%; U = 1,000 → ≥ 94.5%; U = 3,000 → ≥ 83%. So Visium HD 2 µm ≈ 99.9–99.99% zeros, 8 µm ≈ 98–99.5%, 16 µm and segmented cells (~100–1,400 UMIs) ≈ 92–99.5%, classic 55 µm Visium spots (thousands of UMIs) lower. Exact medians are dataset-dependent and were **not** found in a primary source.
- 8 µm = sum of 16 squares and 16 µm = sum of 64, so mean UMI scales ×16 and ×64 vs 2 µm by construction; sparsity falls sub-linearly because highly expressed genes saturate first.
- Bin-level row magnitude reflects **how many cells / how much tissue** a bin covers (bins straddling cells, tissue gaps with ambient RNA), plus tissue-specific capture; cell-level magnitude reflects cell size (OSTA: cell area ∝ #2 µm bins; immune cells smallest, tumour/structural cells larger) plus segmentation errors. Both are "biological" in a sense, but not cell-state.
- bin2cell's destriping and per-bin rescaling likely yield **non-integer** counts (paper says counts are rescaled; integrality not stated); ENACT's weighted assignments necessarily produce fractional counts. Count-model tools (sctransform, SpaNorm, cell2location, RCTD) may need rounding or the unweighted/naive output.
- Stereo-seq bin50 ≈ 25 µm (500 nm pitch) or ~36 µm (715 nm pitch), scaling from Stereopy's bin20 statement — derived, not stated on the page.

### Gaps
- No primary-source per-dataset medians/zero fractions for 2/8/16 µm Visium HD or Stereo-seq bin1/bin50/cellbin; the 10x tech note PDF could not be parsed. Computing from a public 10x dataset would require a large download (not done; needs user permission).
- Whether bin2cell output is integer, and ENACT's post-assignment normalization, unverified.

## Q4. What input representation do spatial clustering / domain-detection methods use?

### Takeaway
Nearly all popular domain methods consume **log-normalized HVG expression** (STAGATE, GraphST, SpaGCN) or **PCs** derived from it (BayesSpace, BANKSY, SEDR, most deep models); BANKSY is notable for using library-size-scaled counts **without log** then internal scaling + PCA; none of the mainstream domain methods operate on raw counts directly, except count-model-based ones (e.g., CellCharter via scVI — not verified here).

### Cited Findings
- **STAGATE** tutorial: `sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)`, `sc.pp.normalize_total(adata, target_sum=1e4)`, `sc.pp.log1p(adata)` — no scaling; this log-normalized HVG matrix is the GAT input. — [STAGATE DLPFC tutorial](https://stagate.readthedocs.io/en/latest/T1_DLPFC.html)
- **GraphST** `preprocess()`: HVG seurat_v3 3000 → `normalize_total(target_sum=1e4)` → `log1p` → `sc.pp.scale(adata, zero_center=False, max_value=10)`; `get_feature` densifies (`toarray()`) HVG matrix into `obsm['feat']`. — [GraphST preprocess.py](https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/preprocess.py)
- **SpaGCN** tutorial: `prefilter_genes(min_cells=3)`, `prefilter_specialgenes`, `sc.pp.normalize_per_cell(adata)`, `sc.pp.log1p(adata)` ("#Normalize and take log for UMI"); Louvain initialization. — [SpaGCN tutorial](https://github.com/jianhuupenn/SpaGCN/blob/master/tutorial/tutorial.md)
- **BANKSY**: `normalizeCounts(log = FALSE)` ("Normalization to mean library size") → `runBanksyPCA` with scaling of own and neighbour features. — [Banksy README](https://github.com/prabhakarlab/Banksy). OSTA HD example runs BANKSY on log-normalized HVGs → 20 PCs → Leiden. — [OSTA binned](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-bin.html)
- iMeta benchmark: methods run "with default parameters"; BASS and STAGATE "achieved peak accuracy using default preprocessing"; DeepST sensitive to #PCs, GraphST stable; BayesSpace limited to ST/Visium grids. — [iMeta 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12747554/)
- Bhuva 2024 best performers for domains were spatially aware methods (BayesSpace, SpaGCN) on scran or log-raw input. — [Bhuva et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/)

### Inferences
- Since most domain methods compute on log-normalized (sparse, non-negative) HVG matrices or on dense PCs, a CosTaL+ kNN built on the **same** representation is directly comparable. GraphST's `zero_center=False` scaling keeps sparsity and non-negativity (division only) — so even "scaled" input is not necessarily dense.
- Methods ingesting PCs effectively use centred, dense, signed data — cosine there behaves like correlation around the global mean, not like a composition similarity.

### Gaps
- BayesSpace (`spatialPreprocess`: logNormCounts + PCA) and SEDR/CellCharter defaults not verified in this session.

## Q5. Deconvolution methods (cell2location, RCTD) and raw counts

### Takeaway
Deconvolution is the one place where raw counts are mandatory: cell2location explicitly requires untransformed, unnormalized spatial counts; RCTD likewise models raw UMI counts (its README only says "RNA counts"; not re-verified in detail).

### Cited Findings
- "Cell2location needs untransformed unnormalised spatial mRNA counts as input." — [cell2location tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html)
- RCTD spatial input must "measure RNA counts across many genes"; tested on Slide-seq, Visium, MERFISH. — [spacexr README](https://github.com/dmcable/spacexr). RCTD (doublet mode) is used on Visium HD 8/16 µm bins in OSTA (38 cell types at 8 µm; 77.39% of 8 µm bins called singlets) and in Seurat's HD vignette. — [OSTA binned](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-bin.html); [Seurat HD vignette](https://satijalab.org/seurat/articles/visiumhd_analysis_vignette)

### Inferences
- Pipelines keep raw counts alongside normalized layers (Stereopy `raw_checkpoint`, AnnData `.raw`/layers, Seurat `counts`), so CosTaL+ can usually access raw counts when needed.
- Fractional counts from ENACT/bin2cell may conflict with count-likelihood deconvolution.

### Gaps
- RCTD's documented integer requirement not confirmed from its help pages in this session.

## Q6. Output-matrix properties per transformation (for CosTaL+: sparse/dense, sign, row-norm informativeness)

### Takeaway
Only transformations that **preserve zeros** (raw, CPx, log1p(CPx), SCT `counts`/`data`, BANKSY normcounts, GraphST non-centred scaling, likely SpaNorm logPAC) stay sparse and non-negative; anything centred or residual-based (Pearson residuals, z-scores, PCA, SpaNorm "pearson", embeddings) is dense and signed. Row norm carries library size + cell density/region in raw counts (Bhuva: partially biological), composition concentration after CPx scaling, a residual depth/n-genes signal after log1p, and "distance from the global mean" after centring.

### Cited Findings
- Zero-preservation of log1p/normalize_total and densification by centring: Stereopy notes `zero_center=False` keeps sparse computation — [Stereopy Square Bin](https://stereopy.readthedocs.io/en/latest/Tutorials/SquareBin_Clustering.html); GraphST densifies features — [GraphST preprocess.py](https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/preprocess.py); SCT `scale.data` = Pearson residuals — [Seurat SCTransform](https://search.r-project.org/CRAN/refmans/Seurat/html/SCTransform.html) (snippet); Giotto pearson_resid/z-scores stored in `scaled` — [Giotto](https://giottosuite.com/reference/normalizeGiotto.html).
- Library size in spatial data is driven by cell number and region — [Bhuva et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025268/); cell size scales with #2 µm bins — [OSTA segmented](https://bioconductor.org/books/release/OSTA/pages/seq-workflow-visium-hd-seg.html).

### Inferences
Property table (my synthesis; sparsity percentages are the arithmetic bounds from Q3, not measured):

| Transformation (where used) | (a) Sparse? zero fraction | (b) Negative? | (c) Row norm biologically informative? |
|---|---|---|---|
| Raw UMI counts (Space Ranger 2/8/16 µm, cells; Stereo-seq GEF; Slide-seq/Curio beads) | Sparse: 2 µm ≳99.9%; 8 µm ~98–99.5%; 16 µm / cells ~92–99%; 55 µm Visium lower | No (integers ≥0) | Yes but confounded: = library size ∝ cells per bin, cell size, region (biological per Bhuva) + permeabilization/capture/tissue gaps (technical) |
| CPx / normalize_total without log; BANKSY `normcounts` | Same pattern as raw | No | Depth removed in L1; L2 norm ∝ profile concentration (Simpson index). In sparse low-depth rows, sampling makes profiles look concentrated → L2 norm **inversely** tracks depth (N singletons: ‖x‖₂ = 10⁴/√N) |
| log1p(CPx) = Seurat LogNormalize, Scanpy, Stereopy, scran logNormCounts, Giotto "normalized" (log2) | Same pattern as raw (log1p(0)=0) | No | Residual depth/n-genes signal: for N singleton counts ‖x‖₂ = √N·log1p(10⁴/N) ≈ 22 (N=10), 46 (100), 76 (1,000) → rises with depth then plateaus; so partly library size again |
| SCTransform `counts` (corrected) / `data` (log1p corrected) | Sparse | No | Depth equalized toward median; norm mostly composition/n-genes |
| SCTransform `scale.data` = Pearson residuals; Giotto/Scanpy analytic Pearson residuals; SpaNorm "pearson" | **Dense** (zeros → −√μ) | **Yes** | Norm ≈ total squared deviation from expected; depth largely removed; mixes distinctiveness + noise |
| z-scored genes (ScaleData, sc.pp.scale zero_center=True, Giotto scaled, Stereopy scale) | **Dense** | **Yes** | Norm = deviation from mean profile; tends to track n-genes/depth |
| GraphST scale(zero_center=False, max 10) | Sparse | No | As log1p, gene-reweighted |
| PCA / BANKSY PCs / GNN/VAE embeddings (STAGATE, GraphST latent) | **Dense** | **Yes** | Norm = distance from centroid in reduced space; often contains depth leakage in leading PCs |
| SpaNorm logPAC (default) | Likely sparse-ish (needs check: whether PAC maps 0→0) | No (log of non-negative adjusted counts) | Library-size effect removed by design while spatially smooth biology kept |
| SpaNorm meanbio / medbio | Dense (fitted means) | No (means ≥0; if log-scaled, sign depends on offset) | Smooth biological expression level; magnitude = modelled expression |
| bin2cell destriped cell sums; ENACT weighted assignment | Sparse | No (likely non-integer) | Cell size + residual capture |

- **Tanimoto vs cosine (math):** with c = cos θ and r = ‖b‖/‖a‖, T = c·r / (1 + r² − c·r). For r = 1, T = c/(2 − c), monotone in c, so Tanimoto **re-ranks neighbours only through norm ratios**. The denominator is always > 0 for non-zero vectors (‖a‖²+‖b‖²−a·b ≥ (‖a‖²+‖b‖²)/2), so T is defined for signed data with range [−1/3, 1] (min at b = −a).
- Practical consequence: Tanimoto adds most over cosine on raw counts / log-raw / log1p input where norms vary 2–10× across bins (library size, cells per bin, gaps). That variation is **domain-informative but confounded** (Bhuva). On Pearson residuals/z-scores/PCA, norm reflects how atypical a spot is, so Tanimoto would down-weight edges between a typical and an extreme spot — not a depth effect, but not clearly "biology" either.
- Backend: sparse backend fits raw, CPx, log1p, SCT data, BANKSY normcounts, GraphST non-centred scaling, logPAC; dense backend fits PCs, residuals, z-scores, embeddings. At Visium HD 2 µm (≥99.9% zeros, millions of rows) a sparse exact cosine is the only practical raw-input option; most users will instead hand over 8/16 µm bins or cells, or a 20–50-PC dense matrix.

### Gaps
- No measured zero fractions or norm-vs-library-size correlations found for specific Visium HD/Stereo-seq/Slide-seq datasets; values above are bounds and illustrative calculations.
- Whether SpaNorm logPAC preserves exact zeros and the effect of `scale.factor` on sparsity unverified.
