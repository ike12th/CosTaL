# Intensity transformations and normalizations for protein single-cell data (suspension cytometry and spatial proteomics), 2023–2026 practice, with output-matrix properties relevant to CosTaL+

Scope note: "verified" below means I read the primary source or the tool's current source code (GitHub raw files fetched 2026-10-12). Statements that came only from a search-engine summary, not from reading the source, are labelled "(search snippet)". Defaults read from source code are labelled "(source code)". Older defaults (pre-2023) are labelled "older". The own-computation numbers come from the repo's `bench/data/Levine_32dim_notransform.fcs` (265,627 cells × 32 markers). The computation was a small numpy script, and its definitions are given inline so it can be reproduced.

---

## Q1. Suspension cytometry: are cofactors 5 (CyTOF), 150 (flow) and ~6000 (spectral) still standard, and what do current tools do by default?

### Takeaway
Yes for CyTOF. arcsinh(x/5) is still the hard-coded default in the main 2023–2026 tools: CATALYST, Spectre `do.asinh`, diffcyt, pytometry and Cytobank-style pegasus. It is also used in recent benchmarks (CytoNormPy 2024/25, Open Problems cyto batch integration 2026). For conventional flow, 150 survives as a documented recommendation (cyCombine, diffcyt/censcyt, pytometry), but much practice uses logicle/biexponential or channel-specific cofactors. For spectral flow there is no single standard. 6000 (cyCombine's recommendation, Jaimes et al. 2022 in OMIQ) is one published anchor. Per-channel values from about 10 to 10,000 (or 1000–5000 for high-background detectors) are the stated best practice. CosTaL's 5/150/6000 triple is the same as cyCombine's documented recommendation.

### Cited Findings
- cyCombine (Pedersen et al., Nat Commun 2022) states: "recommended cofactors are 5 for CyTOF, 150 for flow cytometry, and 6000 for spectral flow cytometry". It also recommends de-randomizing randomized CyTOF counts. Older (2022) but the package is still maintained and used. — [cyCombine paper, PMC8971492](https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492/)
- CATALYST `prepData()` signature in the current devel branch is `prepData(x, panel, md, features = NULL, transform = TRUE, cofactor = 5, ...)`. It accepts a channel-specific cofactor vector, and transformed values go to `assay(x, "exprs")` (source code). — [CATALYST prepData.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/prepData.R)
- Spectre `do.asinh(dat, use.cols, cofactor = 5, append.cf = FALSE, reduce.noise = FALSE)` computes `asinh(value / cofactor)` and allows a per-column cofactor vector (source code). — [Spectre do.asinh.R](https://github.com/ImmuneDynamics/Spectre/blob/master/R/do.asinh.R)
- Spectre's own documentation, however, says "For mass cytometry data, a co-factor of 15 is suitable for most channels". It suggests a starting value of about 100 for conventional flow ("similar compression results to the default scaling of LSR-II in FlowJo v10"). It advises choosing the cofactor from the upper limit of background: about 100–500 for a 10^2 background and about 1000–5000 for a 10^3 background. Its spectral example uses a cofactor of 5000. — [Spectre: Data transformation](https://immunedynamics.io/Spectre/articles/data_transformation.html)
- diffcyt `transformData`: the default cofactor of 5 is "appropriate for mass cytometry (CyTOF) data" (search snippet). The censcyt manual recommends cofactor = 150 for fluorescence flow (search snippet). — [diffcyt transformData](https://rdrr.io/bioc/diffcyt/man/transformData.html); [censcyt manual](https://bioc.r-universe.dev/censcyt/doc/manual.html)
- pytometry's `normalize_arcsinh` suggests a cofactor of 5 for CyTOF and 150 for flow. pegasus's `arcsinh` "mimics Cytobank", computing arcsinh(x/cofactor) with a default of 5.0 (search snippets). — [pytometry](https://pytometry.readthedocs.io/en/latest/generated/pytometry.tl.normalize_arcsinh.html); [pegasus.arcsinh](https://pegasus.readthedocs.io/en/latest/api/pegasus.arcsinh.html)
- CytoNormPy (Exner et al., Cytometry A 2025; bioRxiv 2024): in its benchmarking, "all data were arcsinh transformed using a cofactor of 5" (search snippet). — [CytoNormPy, Cytometry A](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24953); [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.19.604225v1.full.pdf)
- The Open Problems cyto batch-integration benchmark (v0.0.1-rc3, "in development", 2026 page) requires inputs to be "arc-sinh or logicle" transformed. Its CyTOF dataset is arcsinh with cofactor 5. Its 22-colour mouse-spleen flow dataset is compensated and **logicle**-transformed. Its 28-colour spectral dataset is OLS-unmixed and arcsinh-transformed with an unstated cofactor. — [Open Problems cyto_batch_integration](https://openproblems.bio/benchmarks/cyto_batch_integration)
- Jaimes et al. (Cytometry A 2022; two authors are Cytek employees) ran a 32-marker full-spectrum flow (Cytek Aurora) versus mass cytometry comparison in OMIQ. Scaling was first set so ≥99% of events were on scale. The arcsinh cofactor was then set to **6000** for full-spectrum flow, "to ensure the negative population was unimodal around 0", and to **5** for mass cytometry. Their manual-gating comparison used biexponential display for the flow data (search snippet plus abstract). — [Jaimes et al. 2022, PMC9790709](https://pmc.ncbi.nlm.nih.gov/articles/PMC9790709/)
- Den Braanker et al. (Front Immunol 2021, older) give a spectral workflow. For conventional and spectral flow they recommend an individual cofactor per channel or fluorochrome, "usually between 10 and 10.000". Mass cytometry typically uses a fixed value of 5–15. They describe setting cofactors individually as best practice to avoid artefactual populations from "too much resolution around zero", and they used flowVS for per-fluorochrome cofactors (search snippet). — [den Braanker et al. 2021](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.768113/full)
- CellEngine (2023 blog) notes that Aurora avalanche-photodiode data may need a much larger cofactor "to pull together data that appears spread across 0". As a starting heuristic it suggests setting the cofactor to about half the magnitude of the channel minimum (search snippet). — [CellEngine blog 2023](https://blog.cellengine.com/2023/08/28/scaling-flow-cytometry-data-why-and-how/)
- FlowJo applies logicle (biexponential) by default to digital FCS 3.0 data. Arcsinh is available through "Customize Axis", and FlowJo transforms change display, not stored data (search snippet). — [FlowJo docs](https://docs.flowjo.com/?p=11638)

### Inferences
- The "5 / 150 / 6000" triple is a set of inherited conventions (the long-standing CyTOF 5 and flow 150 defaults that tools copy from one another, plus the cyCombine/OMIQ-era spectral 6000), not three independently validated optima. I did not trace the historical origin of 150. For spectral data, 6000 is defensible only as a dataset-agnostic fallback. The literature consensus is per-channel tuning so that the negative population is unimodal around 0.
- Because spectral (APD, unmixed) channels differ by orders of magnitude in negative spread, a single global cofactor on spectral data implies that some channels are over-compressed and others under-compressed. That changes which markers dominate the row norm (see Q8).

### Gaps
- I could not find OMIQ's, Cytek Cloud's or SpectroFlo's documented default arcsinh cofactor for Aurora/Northern Lights data. OMIQ help-center pages did not surface in search.
- I could not verify FlowJo's default arcsinh cofactor, or what FlowJo writes when exporting "scale values" versus channel values.
- I found no 2023–2026 peer-reviewed benchmark that systematically compares spectral-flow cofactors or transformations for clustering accuracy. The closest is OTflow (2022, Q2).

---

## Q2. Per-channel cofactor estimation and logicle/biexponential: what exists and what is used?

### Takeaway
Per-channel estimation exists (flowVS, OTflow, flowScape, autoLogicle), and current tools accept per-channel cofactor vectors (CATALYST, Spectre). Still, the dominant practice is a fixed cofactor for CyTOF and visual or manual per-channel tuning for flow and spectral data. Logicle remains the default display transform in FlowJo and the default `transformFunction` in FlowSOM's `ReadInput`, and is used for some benchmark flow datasets (Open Problems 2026). I found no newer (2023–2026) automatic cofactor method that has displaced flowVS or OTflow.

### Cited Findings
- flowVS `estParamFlowVS` estimates one arcsinh cofactor per channel for a whole sample collection. It picks the cofactor that minimizes Bartlett's statistic (within-population variance homogeneity). It requires ≥2 peaks per channel (search snippet). Older (2016). — [flowVS estParamFlowVS](https://rdrr.io/bioc/flowVS/man/estParamFlowVS.html)
- OTflow (Folcarelli et al., Cytometry A 2022) chooses per-channel arcsinh cofactors automatically. It uses Bartlett's statistic for multi-peak channels and a Jarque–Bera normality test for single-peak channels, and benchmarks against flowVS and a fixed cofactor of 150. The authors argue that poorly chosen parameters create artefacts such as peak splitting and lead to incorrect immunological conclusions (search snippet). — [OTflow, Cytometry A 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.24491)
- FlowSOM `FlowSOM()` and `ReadInput()` defaults (source code): `compensate = FALSE, transform = FALSE, transformFunction = flowCore::logicleTransform(), scale = FALSE`. The logicle transform is used only if `transform = TRUE`. — [FlowSOM 0_FlowSOM.R](https://github.com/SofieVG/FlowSOM/blob/master/R/0_FlowSOM.R)
- The FlowSOM vignette compensates and then logicle-transforms flow data, and notes that arcsinh is preferred for CyTOF (search snippet). The Python FlowSOM port exposes both LogicleTransformer and AsinhTransformer (search snippet). — [FlowSOM vignette](https://bioc-release.r-universe.dev/FlowSOM/doc/FlowSOM.Rnw); [FlowSOM Python example](https://flowsom.readthedocs.io/en/latest/notebooks/example.html)
- The CATALYST `prepData` documentation includes a "channel-specific transformation" example (`prepData(PBMC_fs, cofactor = cf)` with a named vector), and the cofactors are stored in `int_metadata(sce)$cofactor` (source code). — [CATALYST prepData.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/prepData.R)

### Inferences
- Logicle output is a display coordinate with an offset: data value 0 does not map to output 0. If a logicle-transformed matrix were fed to a cosine kNN, every cell would carry a shared positive offset vector. Cosine similarity would then be inflated for all "all-negative" cells, and the origin would not mean "no signal". This follows from the logicle definition (Parks et al. 2006), which I did not re-fetch; verify against flowCore's `logicleTransform` output for x = 0 before relying on it. For CosTaL+, arcsinh (with arcsinh(0) = 0) is the origin-consistent choice. Logicle input would need re-centering or should be rejected.

### Gaps
- I did not find a 2023–2026 paper introducing a new automatic per-channel cofactor estimator that is widely adopted. A further targeted search (e.g., "Bartlett cofactor 2025", "transformation spectral unmixing residual") might find one.
- I did not verify the exact numeric output of `flowCore::logicleTransform()` at data value 0.

---

## Q3. Negative values: do pipelines clip to 0 or keep them? Where do negatives come from?

### Takeaway
Mainstream CyTOF pipelines keep negatives and apply arcsinh directly. CATALYST, diffcyt, the Bioconductor CyTOF workflow and Spectre all do this, and none clips by default. Clipping or "noise reduction" exists only as opt-in or experimental options. In CyTOF, the negatives come from background subtraction plus the acquisition software's default randomization of integer counts. In flow and spectral data they come from compensation or unmixing spread. For Levine_32dim the negatives are tiny after arcsinh(x/5): they account for about 0.01% of each cell's squared norm. Keeping or clipping them barely changes row norms, but clipping changes density from 100% to 73%.

### Cited Findings
- diffcyt: zeros and small negatives arise in channels with no detected ions. The negatives "are due to background subtraction and randomization of integer count values, which are performed by default by the CyTOF instrument software". arcsinh is approximately linear near zero, so it "can handle zeros or small negative values" (search snippet). — [diffcyt transformData](https://rdrr.io/bioc/diffcyt/man/transformData.html)
- The Bioconductor CyTOF workshop (2017, older) states that arcsinh with cofactor 5 works for negative and zero values, which "do not have to be excluded from the analysis" (search snippet). — [BioC2017 CyTOF workshop](https://www.bioconductor.org/help/course-materials/2017/BioC2017/Day2/Workshops/CyTOF/cytoWorkshop.pdf)
- Papoutsoglou et al. (Cytometry A 2019, older) describe three CyTOF randomization types. Type 1 spreads each integer count x uniformly over (x−1, x), so a zero count becomes negative. Type 2 is Gaussian with σ = 1 (the software default). Type 3 uses a negative half-Gaussian for zeros. Randomization degraded clustering: meta-clustering ARI fell from 0.83 to 0.72 under maximal randomization. The authors recommend using randomization only for visualization. — [Papoutsoglou et al. 2019, PMC7027760](https://pmc.ncbi.nlm.nih.gov/articles/PMC7027760/)
- cyCombine recommends de-randomizing randomized CyTOF counts before transformation. Its batch-corrected values "are capped per-marker to the range of the input", so input negatives can persist in the output. — [cyCombine paper, PMC8971492](https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492/)
- Spectre `do.asinh(..., reduce.noise = FALSE)`: the optional, explicitly "experimental" noise reduction subtracts 1 and replaces values < 0 with draws from N(0, 0.01), citing cytofkit issue #71. The default keeps negatives (source code). — [Spectre do.asinh.R](https://github.com/ImmuneDynamics/Spectre/blob/master/R/do.asinh.R)
- CATALYST `compCytof(x, sm, method = c("nnls", "flow"), ...)` defaults to **NNLS** spillover compensation, solved per cell with `nnls(t(sm), u)`. Its compensated counts are therefore non-negative by construction. The `"flow"` option, which uses flowCore matrix-inversion compensation, can create negatives (source code). — [CATALYST compCytof.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/compCytof.R)
- Own computation on Levine_32dim (raw FCS): 27.00% of entries are < 0, all within [−0.291, 0) (median −0.120). There are no exact zeros, and only 0.11% of non-negative values are integers, consistent with randomized counts. After arcsinh(x/5), negatives lie in [−0.058, 0) (median −0.024). The per-cell fraction of negative entries has median 0.25 (p10 0.125, p90 0.44). The median share of a cell's squared L2 norm contributed by negative entries is 0.0001. — [repo data: bench/data/Levine_32dim_notransform.fcs](../../bench/data/Levine_32dim_notransform.fcs)

### Inferences
- For CyTOF data like Levine, negatives are near-zero noise around "no signal". Clipping them to 0, or using `max(x,0)` before arcsinh, changes norms negligibly but turns the matrix 27% sparse. It also removes small, random, sign-flipping noise that can slightly perturb cosine between low-signal cells. Either choice is defensible for CosTaL+. The bigger issue is to document the choice and keep it deterministic.
- For conventional or spectral flow after compensation or unmixing, negative values can be large in raw units (hundreds to thousands), with a symmetric spread around 0 for negative populations. After arcsinh with a cofactor tuned to make that population unimodal around 0, the negative entries are not negligible. They contribute real noise to the norm and to the cosine sign structure. Clipping there would distort the negative population by piling mass at 0 (inference, not measured here).

### Gaps
- I could not verify whether the current Standard BioTools CyTOF XT/Helios acquisition software still randomizes by default, or which randomization type it uses. I also could not determine why Levine's negatives span only [−0.29, 0) rather than (−1, 0).
- I found no 2023–2026 benchmark that directly tests "clip negatives to 0" versus "keep negatives" for clustering accuracy.

---

## Q4. Scaling before clustering and batch normalization in suspension cytometry (FlowSOM, PhenoGraph, CATALYST; CytoNorm 2.0, cyCombine)

### Takeaway
Standard CyTOF and flow clustering pipelines cluster on unscaled arcsinh values. FlowSOM has `scale = FALSE` by default, Spectre's `run.flowsom` passes `scale = FALSE`, and CATALYST `cluster()` feeds `assay(x, "exprs")` directly to FlowSOM. Percentile 0–1 scaling (e.g., 1st/99th) is used for heatmap visualization, not clustering, in CATALYST. Batch normalization operates on the arcsinh scale and returns values on that scale. CytoNorm/CytoNormPy uses per-cluster quantile-spline mapping, now without required controls in CytoNorm 2.0 (2025). cyCombine uses z-score or rank only as an intermediate for SOM co-clustering, then applies ComBat per cluster on the original expression scale. Neither returns z-scored data.

### Cited Findings
- FlowSOM `FlowSOM()` defaults to `scale = FALSE` (with `scaled.center = TRUE, scaled.scale = TRUE` used only if scale is turned on) (source code). — [FlowSOM 0_FlowSOM.R](https://github.com/SofieVG/FlowSOM/blob/master/R/0_FlowSOM.R)
- Spectre `run.flowsom()` calls `FlowSOM::ReadInput(dat.ff, transform = FALSE, scale = FALSE, ...)` (source code). — [Spectre run.flowsom.R](https://github.com/ImmuneDynamics/Spectre/blob/master/R/run.flowsom.R)
- CATALYST `cluster()` runs `ReadInput(flowFrame(t(assay(x, "exprs"))))` then `BuildSOM(..., colsToUse = features)` on markers of class "type", followed by ConsensusClusterPlus on SOM codes with Euclidean distance. No scaling step is applied (source code). — [CATALYST cluster.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/cluster.R)
- CATALYST `plotExprHeatmap(..., scale = c("first", "last", "never"), q = 0.01, ...)`: with scaling on, values are scaled to [0,1] after trimming the lower and upper q quantiles. This is visualization only (source code). — [CATALYST plotExprHeatmap.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/plotExprHeatmap.R)
- CytoNorm 2.0 (Quintelier et al., Cytometry A 2025, 107(2):69–87, doi:10.1002/cyto.a.24910) extends CytoNorm (2020). It can run "without the need for technical replicates or controls", lets the goal distribution "be tailored toward the experimental design", and gives guidance on marker choice for its internal FlowSOM step. — [CytoNorm 2.0 record](https://pure.eur.nl/en/publications/cytonorm-20-a-flexible-normalization-framework-for-cytometry-data/)
- Original CytoNorm (2020, older): it clusters with FlowSOM, then learns per-cluster, per-marker spline transformations that align quantiles of control samples across batches (search snippet). — [CytoNorm 2020, PMC7078957](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7078957/)
- CytoNormPy fully implements CytoNorm 2.0 features: normalization without controls, normalization toward another model or goal distribution, and normalization of markers not used in clustering (search snippet). — [CytoNormPy, Cytometry A 2025](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24953)
- cyCombine: "the expression of every marker is either Z-score normalized or converted to ranks, individually for each batch". "Z-scoring is appropriate for similar datasets", while ranks suit less similar datasets. A SOM is trained on this normalized data, labels are transferred back "to the original expression value cells", ComBat is run per SOM cluster, and outputs are capped to the input range. — [cyCombine paper, PMC8971492](https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492/)
- Open Problems cyto batch integration (2026, in development, release candidate) compares about 20 methods: CytoNorm and cyCombine variants, BatchAdjust, ComBat, GaussNorm, Harmonypy, limma, and Seurat RPCA. It uses nine metrics, including iLISI/cLISI, EMD, FlowSOM mapping similarity and functional-marker preservation. Its QC flags 11 high-severity issues and missing results, so its leaderboard is provisional. — [Open Problems cyto_batch_integration](https://openproblems.bio/benchmarks/cyto_batch_integration)
- A 2024 conference poster (Teiko) on 68 melanoma PBMC samples reports that cyCombine reduced EMD by 0.70 versus 0.37 for CytoNorm, with both preserving biological variance by MAD (search snippet; not peer-reviewed). — [Teiko poster](https://teiko.bio/publication/batch-normalization-mass-cytometry)

### Inferences
- Since FlowSOM, CATALYST and Spectre cluster on unscaled arcsinh data, the de facto practice gives markers unequal weight in the distance, proportional to their arcsinh dynamic range. This is the same convention CosTaL uses, so CosTaL+ defaults are aligned with mainstream practice.
- CytoNorm-style per-marker monotone spline mapping and cyCombine's per-cluster ComBat change absolute marker levels batch-wise. The row norm after batch correction therefore reflects abundance on the "goal batch" scale. It is still an abundance measure, but no longer an instrument-native one.

### Gaps
- I did not find documentation of whether the original Levine/PhenoGraph (2015) analysis scaled markers before kNN, and did not verify that from the primary paper.
- I did not read the CytoNorm 2.0 full text, so how it handles negative values and its transform assumptions (e.g., whether it requires arcsinh input) are unverified.
- There is no independent peer-reviewed 2024–2026 head-to-head of CytoNorm 2.0 versus cyCombine. Open Problems is the closest but incomplete.

---

## Q5. Imaging mass cytometry (IMC): steinbock / CATALYST compensation / imcRtools pipeline — measurement, cofactor, clipping, scaling

### Takeaway
The current Bodenmiller pipeline (steinbock, then imcRtools/IMCDataAnalysis book) works in four steps:
1. It measures **mean** pixel intensity per cell (steinbock default `--aggr mean`).
2. It applies **NNLS** spillover compensation (CATALYST `compCytof` default), which keeps counts non-negative.
3. It transforms with **asinh(counts/1)**, i.e. cofactor 1.
4. It clusters directly on that `exprs` assay without scaling, or on a batch-integrated embedding.

The default fastMNN integration L2-normalizes each cell (`cos.norm = TRUE`), which removes per-cell magnitude. The independent OPTIMAL benchmark (Cytometry A 2024) also found cofactor 1 optimal for IMC and recommends z-scoring after arcsinh to remove batch effects. 99th-percentile handling appears mainly at the pixel or image level for visualization or segmentation, not as a default single-cell step in the current book.

### Cited Findings
- steinbock measurement docs: "By default, pixels belonging to an object are aggregated by taking the mean". `--aggr median` and other numpy functions are optional. — [steinbock measurement.md](https://github.com/BodenmillerGroup/steinbock/blob/main/docs/cli/measurement.md)
- IMCDataAnalysis book, "Read data": "The summarized pixel intensities per channel and cell (here mean intensity) are stored in the `counts` slot". The book applies `asinh(counts(spe)/1)`: "We therefore apply a `cofactor` of `1`", because IMC data "usually display much lower counts" than CyTOF, where 5 is common. It suggests testing alternatives such as `log(counts + 0.01)`. CellProfiler-pipeline means must be rescaled by 65535. — [IMCDataAnalysis: read-data](https://bodenmillergroup.github.io/IMCDataAnalysis/read-data.html)
- IMCDataAnalysis spillover chapter: `compCytof(spe, sm, transform = TRUE, cofactor = 1, isotope_list = isotope_list, overwrite = FALSE)`. With no `method` argument, this uses CATALYST's default `"nnls"`. — [IMCDataAnalysis: spillover](https://bodenmillergroup.github.io/IMCDataAnalysis/spillover-correction.html); [CATALYST compCytof.R](https://github.com/HelenaLC/CATALYST/blob/devel/R/compCytof.R)
- IMCDataAnalysis phenotyping chapter: Rphenograph (k = 45) and scran SNN-Louvain both use `t(assay(spe, "exprs")[use_channel, ])`, i.e. unscaled asinh(x/1). Integrated clustering uses `use.dimred = "fastMNN"`. Heatmaps use `scale = "none"`, with max-scaled and z-scored heatmaps shown only as visualization alternatives. — [IMCDataAnalysis 08-phenotyping.Rmd](https://github.com/BodenmillerGroup/IMCDataAnalysis/blob/main/08-phenotyping.Rmd); [09-singlecell_visualization.Rmd](https://github.com/BodenmillerGroup/IMCDataAnalysis/blob/main/09-singlecell_visualization.Rmd)
- The IMCDataAnalysis batch-correction chapter runs `fastMNN(spe, batch = patient_id, assay.type = "exprs", ...)`, Harmony on PCA of the asinh counts, and Seurat integration with `ScaleData`, i.e. per-feature z-scoring. — [IMCDataAnalysis 07-batch_correction.Rmd](https://github.com/BodenmillerGroup/IMCDataAnalysis/blob/main/07-batch_correction.Rmd)
- batchelor `fastMNN(..., cos.norm = TRUE, ...)`: "cosine normalization should be performed on the input data prior to PCA" is the default (source code). — [batchelor fastMNN.R](https://github.com/LTLA/batchelor/blob/master/R/fastMNN.R)
- OPTIMAL (Hunter et al., Cytometry A 2024, 105(1):36–53) measured per-cell **mean** intensity and titrated the arcsinh cofactor from about 0.1 to 150, using a Fisher discrimination ratio between gated low and high populations. "A value of '1' was optimal". Z-score normalization after arcsinh(cofactor 1) was "used to remove batch effect as well as to normalize marker intensities relative to one another". Hot pixels were removed by "capping at the top and bottom 5%". FlowSOM (27/30 clusters annotatable) beat PhenoGraph (18/30). — [OPTIMAL, PMC10952805](https://pmc.ncbi.nlm.nih.gov/articles/PMC10952805/)
- A 2024 IMC segmentation study normalized biomarker channels by dividing each image by its 99th-percentile intensity (pixel level) "to mitigate the influence of any extreme pixel values" (search snippet). — [arXiv 2402.04446](https://arxiv.org/pdf/2402.04446)

### Inferences
- IMC output after the book's default (NNLS → mean → asinh(x/1)) is **non-negative**. Its density depends on panel and tissue: a per-cell mean is exactly 0 only when every pixel in the cell has 0 counts, which is plausible for low-abundance channels in small cells. Expect partial sparsity, but I did not measure it.
- Cofactor 1 makes asinh close to log(2x) for x ≳ 2. The row norm then behaves like a log-scale abundance vector, compressing high-expressing markers strongly. Magnitude remains informative, but is closer to "how many markers are above background" than to total ion load (Q8 Levine analogue: rho = 0.54 versus raw sum under cofactor 1).
- Any IMC workflow that clusters on the fastMNN embedding has already removed per-cell magnitude (cos.norm). Tanimoto weighting on such an embedding is a monotone function of cosine and adds nothing (Q8).

### Gaps
- I could not verify the exact preprocessing of the classic histoCAT-era IMC papers (e.g., Jackson et al. 2020 Nature: 99th-percentile censoring and 0–1 scaling before PhenoGraph). These are often cited as the origin of "99th percentile censoring", but their methods text was not retrievable, so I treat this as unverified and older.
- No measured sparsity numbers for IMC per-cell mean matrices.

---

## Q6. MIBI, CODEX/PhenoCycler, CyCIF/MCMICRO and similar: what transformations do current tools use?

### Takeaway
Per-cell quantification is almost universally **mean intensity**, or equivalently total intensity divided by cell area: MCMICRO mcquant, steinbock, spatialproteomics, ark `size_norm`, UniFORM. After that the lineages diverge:
- **MIBI (Angelo lab ark):** total counts → divide by cell size → arcsinh(100·x), which is cofactor 0.01. Pixie adds per-pixel sum-to-1 normalization and 99.9th-percentile per-marker scaling, which deliberately removes magnitude.
- **CODEX (Nolan lab / SPACEc):** per-marker **z-score** by default, inherited from the Hickey 2021 benchmark. Arcsinh(x/150), 1–99% min-max and "double z-score" (with a per-cell row z) are options.
- **CyCIF (MCMICRO/scimap):** mean intensity → log1p → GMM- or manual-gate-anchored piecewise min-max to [0,1] per image, with the gate at 0.5.
- **spatialproteomics (Nat Methods 2026):** default `arcsinh(x/5)`.

### Cited Findings
- **ark-analysis (MIBI), source code:** `compute_marker_counts(..., extraction = 'total_intensity')`, then `transform_expression_matrix(transform = 'size_norm')` divides each channel by cell size, then `transform = 'arcsinh'` multiplies by `linear_factor` (default 100) and applies `np.arcsinh`. The two outputs are `cell_table_size_normalized` and `cell_table_arcsinh_transformed`. — [ark segmentation_utils.py](https://github.com/angelolab/ark-analysis/blob/main/src/ark/segmentation/segmentation_utils.py); [ark marker_quantification.py](https://github.com/angelolab/ark-analysis/blob/main/src/ark/segmentation/marker_quantification.py)
- **Pixie (Liu et al., Nat Commun 2023), Angelo lab:**
  - Pixel clustering: Gaussian blur (σ = 2), then "pixel normalized by dividing by the total pixel expression" so each pixel sums to 1, then "each marker is normalized by its 99.9th percentile value". The stated rationale is that absolute pixel intensity varies with "drifts in instrument sensitivity or variation in tissue fixation and staining" and "we are interested in the ratio between the phenotypic markers".
  - Removing either normalization significantly worsened cluster consistency.
  - Stated limitation: Pixie "cannot detect phenotypes characterized by differential expression of the same marker".
  - Cell clustering uses pixel-cluster counts "normalized by the total cell size" plus 99.9% feature normalization. It reached F1 0.90 versus 0.66 for integrated expression against manual annotation.
  — [Pixie, PMC10393943](https://pmc.ncbi.nlm.nih.gov/articles/PMC10393943/)
- Pixie extended methods (2025 protocols.io) describe 99.9th-percentile normalization before pixel clustering so very bright channels do not dominate (search snippet). — [protocols.io Greenwald/Nederlof extended methods](https://www.protocols.io/view/greenwald-nederlof-et-al-extended-methods-hhqwb35xf.html)
- **SPACEc (Nolan-lab CODEX/PhenoCycler pipeline), source code:** `format(data, list_out, list_keep, method = "zscore", ArcSin_cofactor = 150)`. The options are:
  - `"zscore"`: per-marker, `zscore(df, axis 0)`, the default.
  - `"ArcSin"`: `arcsinh(x/150)`.
  - `"MinMax"`: per marker, clip to the 1st/99th quantile, then scale to [0,1].
  - `"double_zscore"`: per-marker z, then **per-cell** z (axis 1), then −log(1 − Φ(z)).
  — [SPACEc _general.py](https://github.com/yuqiyuqitan/SPACEc/blob/master/src/spacec/preprocessing/_general.py)
- **spatialproteomics (Nature Methods 2026), source code:**
  - `add_quantification(func = "intensity_mean")`.
  - `transform_expression_matrix(method = "arcsinh", cofactor = 5.0, ...)`, with options `"zscore"` (per channel), `"minmax"` (per channel, global min/max, no clipping), `"double_zscore"` (same as SPACEc) and `"clip"` (percentile 1–99, computed over the whole matrix).
  — [spatialproteomics pp/utils.py](https://github.com/sagar87/spatialproteomics/blob/main/spatialproteomics/pp/utils.py); [spatialproteomics pp/preprocessing.py](https://github.com/sagar87/spatialproteomics/blob/main/spatialproteomics/pp/preprocessing.py); [Nature Methods 2026 article](https://www.nature.com/articles/s41592-026-03155-1)
- **MCMICRO quantification (mcquant):** "By default only mean intensity is calculated". `intensity_median` and `intensity_sum` are optional. — [labsyspharm/quantification README](https://github.com/labsyspharm/quantification)
- **scimap `pp.rescale`** (CyCIF/MCMICRO ecosystem) defaults are `log = True, method = 'all', gmm_components = 3`.
  - Raw data are `log1p`-transformed.
  - A GMM gate is placed between the two highest components, after clipping to the 0.01–99.99th percentiles for fitting only. Manual gates can be supplied instead.
  - Values are scaled **per image**, piecewise: below the gate → (0, 0.5), above → (0.5, 1). "values below 0.5 are considered non-expressing".
  - `adata.X` is replaced.
  — [scimap rescale](https://scimap.xyz/Functions/pp/rescale/)
- **Tribus (Bioinformatics 2025)** preprocesses multiplexed-imaging data with log and z-score normalization, 99.9th-percentile outlier removal, and arcsinh with cofactor 5 in one dataset (search snippet). — [Tribus, Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/3/btaf082/8029662)
- **SpatialSort (2023):** MIBI DLBCL data were arcsinh-transformed "and divided by 10" to put them on a CyTOF-like scale (search snippet). — [SpatialSort, PMC10311307](https://pmc.ncbi.nlm.nih.gov/articles/PMC10311307/)
- A 2025 colorectal multiplexed-imaging study scaled each channel by its 99.9th percentile, multiplied by 10, applied arcsinh, and divided by arcsinh(10), keeping 99.9% of values in [0,1] (search snippet). — [arXiv 2510.05083](https://arxiv.org/pdf/2510.05083)
- **UniFORM (Cell Reports Methods 2025)** operates on per-cell **mean** intensities and applies a log transform during histogram computation (details in Q7). — [UniFORM, PMC12539234](https://pmc.ncbi.nlm.nih.gov/articles/PMC12539234/)

### Inferences
- MIBI via ark: arcsinh(100 · mean counts per pixel) is effectively a very small cofactor (0.01). MIBI per-pixel means are small fractions, so this is close to log(200·x) for most positive cells. Raw ion counts are ≥ 0 and the transform keeps 0 at 0, so the output is non-negative, with exact zeros wherever a cell has no counts in a channel. MIBI is a counting technology with sparse pixels, so the per-cell matrix is likely partially sparse; I did not measure this.
- CODEX/CyCIF fluorescence: per-cell means after background subtraction (if the vendor processor subtracts blanks) can be ≤ 0, which is why Hickey 2021 motivated arcsinh "for low or negative values resulting from background subtraction". SPACEc's default z-score output is dense and about half negative.
- Pixie's sum-to-1 normalization is the extreme case of a per-row normalization. The authors deliberately discard magnitude, which is the information Tanimoto exploits, and acknowledge that this loses phenotypes defined by expression level.

### Gaps
- I did not verify the Akoya PhenoCycler/QuPath or CODEX Processor default per-cell statistic (mean versus total) or its background-subtraction behaviour.
- I did not verify Keren et al. 2018 (MIBI TNBC) preprocessing, which is older; the current ark code is used instead.
- I did not verify the CyCIF lab's current recommended pipeline (e.g., CyLinter QC then scimap rescale) beyond scimap defaults.

---

## Q7. What do 2021–2026 benchmarks of multiplexed-imaging normalization conclude?

### Takeaway
The evidence is thin and somewhat contradictory:
- Hickey et al. (2021, CODEX) found normalization choice mattered more than the clustering algorithm and recommended per-marker z-score.
- OPTIMAL (2024, IMC) recommended arcsinh(cofactor 1) followed by z-score.
- UniFORM (2025, CyCIF/ORION/COMET), a developer benchmark, found z-score, ComBat and MxNorm-registration distort marker relationships in right-skewed data. In that benchmark, simple per-image mean division was the runner-up and UniFORM (negative-peak landmark registration in log space) was best.
- Pixie (2023, MIBI) showed per-pixel sum normalization plus 99.9th-percentile scaling improved pixel-cluster consistency.

I found no 2023–2026 study that isolates "arcsinh only" versus "arcsinh + z-score" on ground-truth labels.

### Cited Findings
- **Hickey et al. (Front Immunol 2021, CODEX, older but the basis of SPACEc defaults)** compared five normalizations with four clustering algorithms:
  - The normalizations were none; per-marker Z; "log (double Z)" (per-marker Z then per-cell Z, then −log(1−p)); 1st/99th-percentile min-max; and arcsinh with cofactor 150.
  - Results: "Z-score normalization was generally effective in mitigating the effects of noise" and "best reduced high background signal noise". Neuroendocrine-cell F-score was 0.65 with Z versus 0 with no normalization. Min-max and arcsinh were more consistent for neutrophils.
  - Recommendation: "Z normalization of values from each marker prior to cell-type identification".
  — [Hickey et al. 2021, PMC8415085](https://pmc.ncbi.nlm.nih.gov/articles/PMC8415085/)
- **OPTIMAL (Cytometry A 2024, IMC):** cofactor 1 maximized the Fisher discrimination between negative and positive signal, and z-score after arcsinh removed batch effects. — [OPTIMAL, PMC10952805](https://pmc.ncbi.nlm.nih.gov/articles/PMC10952805/)
- **UniFORM (Cell Reports Methods 2025; bioRxiv Dec 2024):**
  - Method: aligns the negative (leftmost-mode) population of each marker across samples by rigid landmark registration on log-transformed histograms, at the feature (per-cell mean) or pixel level.
  - Datasets: CRC-ORION (6 slides, 1 batch), PRAD-CyCIF (20 samples, 7 batches) and TMA-Lunaphore COMET (3 batches).
  - Average kBET: UniFORM 0.608; mean division 0.542; MxNorm-registration 0.388; Z-score 0.286; ComBat 0.273; raw 0.001.
  - Average silhouette: UniFORM 0.487; ComBat 0.438; raw 0.401; Z-score 0.385; mean division 0.375; MxNorm −0.021.
  - The authors state "MTI data often show inherent right skewness and heterogeneity, making Z score normalization less suitable". They also state "in right-skewed data, Z score normalization causes lower-intensity data points to become overly compressed around zero", and that MxNorm, Z score and ComBat "distort marker relationships, weakening mutual exclusivity".
  - Caveat: this is a developer-run benchmark.
  — [UniFORM, PMC12539234](https://pmc.ncbi.nlm.nih.gov/articles/PMC12539234/)
- **mxnorm (Harris et al., JOSS 2022, older):** an R package implementing and evaluating slide-level normalizations for multiplexed imaging (mean division, log variants, functional-data registration) with user-defined methods and thresholds. — [mxnorm vignette](https://cran.r-project.org/web/packages/mxnorm/vignettes/mxnorm-vignette.html); [mxnorm JOSS](https://joss.theoj.org/papers/10.21105/joss.04180.pdf)
- **Pixie (Nat Commun 2023):** removing pixel normalization yielded "one dominating pixel cluster that was poorly defined". Removing 99.9% normalization yielded "poorly defined clusters that expressed many markers". — [Pixie, PMC10393943](https://pmc.ncbi.nlm.nih.gov/articles/PMC10393943/)
- **Phenotyping-method benchmarks** (Tribus 2025, STARLING, DeepCellTypes/MAPS comparisons) bundle preprocessing with the method and do not isolate the normalization effect (search snippet summary). — [Tribus, Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/3/btaf082/8029662)

### Inferences
- The benchmarks optimize different targets. Hickey optimizes cell-type F-score within one dataset, where z-score's equal marker weighting helps dim lineage markers. UniFORM optimizes cross-batch mixing plus preservation of biological relationships, where z-score's distortion of skewed marker distributions hurts. Pixie optimizes pixel-cluster stability, where removing magnitude helps. None evaluates whether per-cell magnitude carries useful information. For CosTaL+ that question is open, and the safest default is to keep arcsinh-only input, matching CATALYST, steinbock and FlowSOM practice, and treat z-scoring as an explicit opt-in.

### Gaps
- I found no 2023–2026 independent (non-developer) benchmark of multiplexed-imaging normalization across MIBI, IMC, CODEX and CyCIF with ground truth.
- A search-snippet claim that Astir/TYPEx-style CODEX pipelines use arcsinh(x/150) plus 0–99% winsorization could not be tied to a verified source, so I omitted it.

---

## Q8. What each transformation does to the output matrix: (a) sparse vs dense, (b) negative values possible, (c) does the per-cell row norm still mean "overall marker abundance"? Including how z-scoring and min-max change the row norm.

### Takeaway
Only transformations that act **per marker and monotonically, without centering** keep the row norm as an abundance-like quantity. Examples are arcsinh at any cofactor, percentile division, min-max with lower bound at about 0, mean division and per-marker spline batch mapping. The cofactor changes which markers dominate the norm.

- **Per-marker z-scoring** centres each marker. The row norm then becomes the cell's distance from the population-mean profile, i.e. atypicality. In Levine, z-score row norm ranks correlate only ρ = 0.50 with abundance but ρ = 0.82 with distance from the mean. The matrix becomes about 65% negative.
- **Min-max with 1/99 clipping** keeps an abundance meaning (ρ = 0.92) but equal-weights markers and makes everything non-negative.
- **Any per-cell (row) normalization** destroys magnitude: L1/L2 normalization, Pixie's sum-to-1, double z-score's row-z, and fastMNN's `cos.norm`. After L2 normalization, Tanimoto = cos/(2 − cos) is a monotone function of cosine and adds no information.

### Cited Findings
- **Own computation on Levine_32dim** (265,627 cells, 32 markers, raw randomized CyTOF). Definitions:
  - "abundance" = Σ_j asinh(max(x_j, 0)/5).
  - ρ = Spearman correlation of the per-cell L2 norm of the transformed matrix with abundance, with raw positive sum Σ max(x, 0), and with the asinh(x/5) row norm.
  - Percentiles are per marker on asinh(x/5).

  | Transform | density | frac < 0 | ρ(norm, abundance) | ρ(norm, raw sum) | ρ(norm, asinh5 norm) |
  |---|---|---|---|---|---|
  | asinh(x/5), keep negatives | 1.000 | 0.270 | 0.790 | 0.762 | 1.000 |
  | asinh(max(x,0)/5), clip | 0.730 | 0.000 | 0.790 | 0.762 | 1.000 |
  | asinh(x/1) | 1.000 | 0.270 | 0.968 | 0.544 | 0.913 |
  | asinh(x/150) | 1.000 | 0.270 | 0.297 | 0.992 | 0.722 |
  | asinh(x/5) → per-marker z-score | 1.000 | 0.651 | 0.502 | −0.061 | 0.318 |
  | asinh(x/5) → 1/99-pct clip + min-max [0,1] | 0.990 | 0.000 | 0.919 | 0.456 | 0.834 |
  | asinh(x/5) → ÷ 99th pct, censor at 1 | 1.000 | 0.270 | 0.921 | 0.418 | 0.818 |
  | asinh(x/5) → per-cell L1 normalization | 1.000 | 0.270 | **−0.783** | 0.173 | −0.286 |

  Also: for z-scored data, ρ(‖z‖, ‖A − mean(A)‖) = 0.820. The asinh(x/5) row norm spans p1 6.35 to p99 12.63 (CV 0.122), so magnitude varies about 2× across cells. — [repo data: bench/data/Levine_32dim_notransform.fcs](../../bench/data/Levine_32dim_notransform.fcs)
- Pixie's authors state that per-pixel sum normalization means the method "cannot detect phenotypes characterized by differential expression of the same marker". This is direct evidence that row normalization removes level information. — [Pixie, PMC10393943](https://pmc.ncbi.nlm.nih.gov/articles/PMC10393943/)
- The double z-score in SPACEc and spatialproteomics applies `zscore(..., axis=1)` per cell after the per-marker z, then −log(1 − Φ(z)) (source code). — [SPACEc _general.py](https://github.com/yuqiyuqitan/SPACEc/blob/master/src/spacec/preprocessing/_general.py); [spatialproteomics pp/utils.py](https://github.com/sagar87/spatialproteomics/blob/main/spatialproteomics/pp/utils.py)
- fastMNN's default `cos.norm = TRUE` L2-normalizes cells before PCA, and the IMC book's integrated clustering runs on this embedding. — [batchelor fastMNN.R](https://github.com/LTLA/batchelor/blob/master/R/fastMNN.R); [IMCDataAnalysis 08-phenotyping.Rmd](https://github.com/BodenmillerGroup/IMCDataAnalysis/blob/main/08-phenotyping.Rmd)
- UniFORM's authors report that z-scoring right-skewed MTI data "causes lower-intensity data points to become overly compressed around zero" and distorts mutual exclusivity. — [UniFORM, PMC12539234](https://pmc.ncbi.nlm.nih.gov/articles/PMC12539234/)

### Inferences

**Master table of output-matrix properties.** These are derived from the transform definitions plus the Levine measurements. "Abundance" means per-cell overall marker level.

| # | Transformation (where used) | (a) Sparse or dense | (b) Can be negative? | (c) Does the row norm still mean overall abundance? |
|---|---|---|---|---|
| 1 | arcsinh(x/c), keep negatives. CyTOF c = 5 (CATALYST, Spectre, diffcyt, CosTaL); flow c = 150; spectral c ≈ 6000 or per channel | Same zero pattern as the input. Randomized CyTOF: 100% dense (Levine). | Yes, wherever the input is negative: CyTOF randomization/background (small, ≥ −0.06 after c = 5 in Levine); flow/spectral compensation/unmixing spread (larger). | **Yes.** Small c behaves log-like, so the norm reflects how many markers are on (Levine c = 1: ρ = 0.97 with asinh5 abundance). Large c is near-linear, so the norm is dominated by the brightest channels (c = 150 on CyTOF: ρ = 0.99 with raw sum). |
| 2 | arcsinh with negatives clipped to 0 (opt-in; Spectre `reduce.noise` is a variant that adds noise instead) | Sparse in proportion to the input negative fraction (Levine 73% dense). | No. | Yes, essentially unchanged for CyTOF (negatives are about 0.01% of the squared norm). For flow/spectral the change is larger; inference, not measured. |
| 3 | IMC default: NNLS compensation → per-cell mean → asinh(x/1) (steinbock + CATALYST + IMC book) | Dense to partially sparse; exact zeros only for all-zero cells × channels (unmeasured). | No (NNLS ≥ 0, mean ≥ 0). With the `"flow"` compensation method, yes. | Yes, on a near-log scale: compresses bright markers and emphasizes the count of above-background markers. |
| 4 | MIBI ark: total counts / cell area → arcsinh(100·x) | Likely partially sparse (count data); unmeasured. | No. | Yes: abundance per unit area on a near-log scale. |
| 5 | Per-marker z-score (SPACEc default; Hickey 2021 recommendation; OPTIMAL post-arcsinh; Seurat `ScaleData`; cyCombine intermediate) | Dense (zeros only by coincidence). | Yes, about 50–65% of entries (Levine 65%). | **No.** The norm becomes diagonal-Mahalanobis distance from the mean cell, i.e. atypicality (Levine ρ = 0.50 with abundance, 0.82 with distance from mean). A uniformly dim cell and a uniformly bright cell both get large norms with opposite signs, so cosine treats them as anti-correlated. |
| 6 | Per-marker min-max with 1/99-pct clip to [0,1] (SPACEc `MinMax`; Hickey; CATALYST heatmaps only) | Nearly dense (zeros = cells at or below the 1st percentile; Levine 99% dense). | No. | **Mostly yes, re-weighted.** Each marker contributes in units of its own dynamic range, so the norm is "sum of relative marker levels" (Levine ρ = 0.92). If done per batch or image, it is relative to that batch or image. Unclipped global min-max (spatialproteomics `minmax`) is sensitive to outliers. |
| 7 | ÷ 99th (or 99.9th) percentile per marker, censor at 1 (IMC/MIBI practice; Pixie's 99.9% step) | Same zero pattern as the input. | Only if the input was negative. | Yes, re-weighted as in row 6 (Levine ρ = 0.92). |
| 8 | Per-image mean division (mxnorm; UniFORM runner-up) | Same zero pattern as the input. | Only if the input was negative. | Yes, relative to the image average per marker. Within an image, ranking by norm is an abundance ranking with markers equal-weighted by their mean. |
| 9 | scimap rescale: log1p → per-image gate-anchored piecewise min-max, gate → 0.5 (CyCIF/MCMICRO) | Nearly dense (0 only at the per-image minimum). | No (range [0,1]). | Partly. The norm reflects how many markers are above gate and by how much, equal-weighted per marker and relative per image. Absolute intensity is lost and the piecewise mapping is non-linear around the gate. |
| 10 | Double z-score: per-marker z → per-cell z → −log(1−Φ) (Hickey "log double Z"; SPACEc/spatialproteomics option) | Dense. | No (−log(1−Φ) > 0). | **No.** The per-cell z removes each cell's mean and scale, so the norm reflects profile shape only. |
| 11 | Per-cell L1/L2 normalization (Pixie pixel sum-to-1; fastMNN `cos.norm`; any "cosine-normalized" input) | Same zero pattern as the input. | Only if the input was negative. | **No.** L2 gives a constant norm. After L1 normalization the L2 norm measures concentration in a few markers, which in Levine anti-correlates with abundance (ρ = −0.78). |
| 12 | Pixie cell features: pixel-cluster counts / cell size (+ 99.9% normalization) | Sparse (most cells contain few pixel clusters). | No. | No for protein abundance: the vector is composition (row sum ≤ 1 before 99.9% scaling). |
| 13 | Logicle/biexponential (FlowJo default display; FlowSOM `transformFunction` default; Open Problems flow dataset) | Dense. | Typically non-negative on the display scale; data 0 maps to a positive offset (unverified, Q2). | Biased: the offset adds a common vector to every cell, so the norm is inflated for negative cells and cosine is distorted. Needs re-centering. |
| 14 | cyCombine output (ComBat per SOM cluster on the arcsinh scale, capped to the input range) | Dense. | Yes, if the input had negatives. | Yes: abundance on the batch-corrected arcsinh scale. |
| 15 | CytoNorm / CytoNorm 2.0 (per-cluster, per-marker monotone splines to a goal distribution) | Same as input. | Same as input (unverified). | Yes: abundance mapped to the goal-batch scale. |
| 16 | UniFORM (log-space shift aligning negative peaks per marker per sample) | Dense. | Unknown/unverified (output float64, log-domain shift). | Probably yes: a per-marker shift preserves within-sample ordering; inference. |

**Implications for CosTaL+:**
- Tanimoto T = a·b / (‖a‖² + ‖b‖² − a·b). Write cos = a·b / (‖a‖‖b‖) and r = ‖a‖/‖b‖. Then T = cos / (r + 1/r − cos). T is a function of cosine and the norm ratio only.
  - When r ≡ 1 (rows 10–11 and any cos-normalized embedding), T = cos/(2 − cos), a strictly monotone function of cos. Tanimoto then adds no information beyond cosine.
  - On per-marker z-scored input (row 5), r compares "atypicality" rather than abundance. Tanimoto would down-weight edges between a near-average cell and an extreme cell, which is a different biological statement from CosTaL's "abundance-aware" intent.
  - On rows 1–4, 6–8 and 14–15, r is an abundance ratio, consistent with CosTaL's design.
- Sign handling for CosTaL+: on CyTOF arcsinh input, negatives are numerically negligible for norms (Levine: about 0.01% of squared norm). Keeping or clipping is a density/performance decision (100% versus 73% dense), not a semantic one. On z-scored input, negatives are half the data and carry the meaning, so they must never be clipped. On flow/spectral arcsinh input, negatives are larger and symmetric around 0 for negative populations, so clipping would bias them.
- A practical guard for CosTaL+ would be to detect probable z-scored input (per-column mean ≈ 0, sd ≈ 1) or row-normalized input (row norms constant, or row sums constant). It could then warn that Tanimoto re-weighting is either redundant (row-normalized) or semantically changed (z-scored). This is a design suggestion, not a literature finding.

### Gaps
- Sparsity of real IMC, MIBI and CODEX per-cell matrices was not measured. No public matrix was loaded here, and the zero fractions in the table for those platforms are inferences.
- The logicle offset claim (row 13) and the UniFORM output sign (row 16) were not verified from source code.
- No published study evaluates whether per-cell magnitude (row norm) improves cell-type recovery in protein data. CosTaL (2023) itself is the closest, and its benchmark results were not re-checked here.
