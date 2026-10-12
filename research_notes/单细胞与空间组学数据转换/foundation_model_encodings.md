# Single-cell / spatial foundation models: input value encodings and output cell-embedding properties (2023–2026), with relevance to cosine-kNN + Tanimoto weighting (CosTaL+)

Research date: 2026-10-12. Sources were fetched in this session unless marked "(not re-verified this session)". Code-level claims come from the official repositories' source files as of the fetch date. Inferences are kept separate from cited findings.

---

## Q1. What input transformation does each model apply to expression values, and what normalization comes first?

### Takeaway
There are four families of input encoding. (1) **Per-cell rank encodings**: Geneformer, Nicheformer, State SE's top-2048 ordering, and, in effect, scGPT's per-cell quantile bins. These throw away absolute magnitude and library size inside the cell. (2) **Continuous value projection** through an MLP on log-transformed values: scPRINT, CellFM, and scFoundation/AIDO.Cell via "auto-discretization". scFoundation, scPRINT and AIDO.Cell also add an explicit read-depth signal. (3) **Expression-weighted gene sampling**: UCE. (4) **Raw-count generative modeling**: TranscriptFormer. Standard preprocessing is per-cell total-count normalization to 1e4 followed by log1p, but several models want raw counts and normalize internally.

### Cited Findings

**Geneformer (Theodoris et al., Nature 2023; V2 Dec 2024)**
- Rank value encoding. The model card says "genes are ranked by their expression in that cell scaled by their expression across the entire Genecorpus" — [Geneformer HF model card](https://huggingface.co/ctheodoris/Geneformer). The specific recipe is: normalize by the cell's total counts, divide by each gene's nonzero median across the pretraining corpus, then rank. This comes from the original Nature paper (not re-verified this session); the model card only says "scaled".
- Input length: 2,048 for V1-10M and 4,096 for V2-104M and V2-316M. V2 was trained on about 104M transcriptomes with a vocabulary of about 20K protein-coding genes. The default checkpoint is V2-316M — [Geneformer HF](https://huggingface.co/ctheodoris/Geneformer). The configs confirm `max_position_embeddings` 2048 for V1 ([V1 config](https://huggingface.co/ctheodoris/Geneformer/blob/main/Geneformer-V1-10M/config.json)) and 4096 for V2-316M ([V2 config](https://huggingface.co/ctheodoris/Geneformer/blob/main/config.json)).
- An independent description by Kedzierska et al. reads: "The input to the Geneformer model is a ranked list where a gene's position represents its expression" — [Kedzierska et al., Genome Biol 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007350/).

**scGPT (Cui et al., Nature Methods 2024)**
- Value binning is per cell and uses quantiles of the cell's nonzero values: `bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))`. Bin 0 is reserved for zeros ("the first bin is always a spectial for zero"). Tied values are randomly spread across adjacent bins by `_digitize` — [scGPT preprocess.py](https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py).
- `Preprocessor` code defaults are `normalize_total=1e4`, `log1p=False` (the docstring wrongly says True), `subset_hvg=False`, `hvg_flavor="seurat_v3"` and `binning=None`, so the caller must set n_bins — [scGPT preprocess.py](https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py).
- The zero-shot embedding path (`embed_data`/`get_batch_cell_embeddings`) does **not** call normalize_total or log1p. It takes `adata.X` directly, drops genes missing from the vocabulary, keeps only nonzero genes, prepends `<cls>`, and bins inside the collator (`do_binning=True`) with `max_length=1200` and `sampling=True` — [scGPT cell_emb.py](https://github.com/bowang-lab/scGPT/blob/main/scgpt/tasks/cell_emb.py).
- One benchmark reports scGPT "Binning (n_bins=51)" — [Hou et al. 2026 bioRxiv](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/).
- **Conflict:** Kedzierska et al. say scGPT bins "into 50 equidistant bins" ([Genome Biol 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007350/)). The code uses per-cell quantile bins, which are equal-frequency, not equidistant — [preprocess.py](https://github.com/bowang-lab/scGPT/blob/main/scgpt/preprocess.py). The code is authoritative.

**scBERT (Yang et al., Nat Mach Intell 2022; pre-2023 but still benchmarked)**
- Data are normalized with `sc.pp.normalize_total` and `sc.pp.log1p`. Expression is discretized into bins (`num_tokens = 7`, "Number of bins in expression embedding"). Model dimension `dim = 200`, with 6 Performer layers and 10 heads — [scBERT GitHub](https://github.com/TencentAILabHealthcare/scBERT).

**scFoundation (Hao et al., Nature Methods 2024) / xTrimoGene**
- Uses a fixed 19,264-gene input list ([scFoundation model README](https://github.com/biomap-research/scFoundation/tree/main/model)).
- Single-cell preprocessing in code: `np.log1p(x / x.sum() * 1e4)`, i.e. normalize to 1e4 then log1p — [get_embedding.py](https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py).
- **Read-depth tokens:** two scalars are appended to the gene vector. The last is always `log10(totalcount)` (the "S", source-depth token). The second-to-last ("T", target depth) depends on `tgthighres`: `t4` (default) gives a constant 4.0, `f` gives log10(total × fold) and `a` gives log10(total) + addend — [get_embedding.py](https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py).
- Option `pre_normalized` (`F`/`T`/`A`) declares whether input is already normalize+log1p; `A` appends total count as column 19,265 — [scFoundation README](https://github.com/biomap-research/scFoundation/tree/main/model).

**AIDO.Cell (GenBio AI, Ho et al., bioRxiv/NeurIPS-W 2024)**
- A dense BERT-style encoder over the whole transcriptome: "capable of handling the entire human transcriptome as input". It "uses an auto-discretization strategy for encoding continuous gene expression values". For pretraining, "a single cell expression is downsampled into a low read depth" and the model predicts higher-depth values of masked genes, i.e. a read-depth-aware objective — [AIDO.Cell-100M model card](https://huggingface.co/genbio-ai/AIDO.Cell-100M).
- Sizes: 3M (6 layers, hidden 128), 10M (8 layers, 256), 100M (18 layers, 650), 650M (32 layers, 1280) — [AIDO.Cell-100M card](https://huggingface.co/genbio-ai/AIDO.Cell-100M).

**UCE — Universal Cell Embeddings (Rosen et al., bioRxiv 2023 → Nature Methods 2024/25)**
- Input `.X` "should be scRNA-seq counts", with gene symbols rather than Ensembl IDs — [UCE GitHub](https://github.com/snap-stanford/UCE).
- Gene tokens are ESM2 protein embeddings; the input is a count matrix plus ESM2 protein embeddings — [Helical UCE model card](https://helical.readthedocs.io/en/latest/model_cards/uce/).
- The released large model is identified as `33l_8ep_1024t_1280` (33 layers, 1,024 tokens, 1,280-dim output) — [Hou et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/).
- Not re-verified this session: the gene-sampling scheme (genes sampled with probability weighted by log expression, then grouped and ordered by chromosome position). This comes from the UCE preprint, which I could not retrieve.

**Nicheformer (Tejada-Lapuerta, Schaar et al., Nature Methods 2025; spatial + dissociated)**
- Cells are first normalized to 10,000 counts. Each gene is then divided by a **technology-specific nonzero mean vector**, computed over nonzero values per assay; all dissociated datasets share one mean vector. Genes are ranked "from most to least expressed", non-expressed genes are dropped, contextual tokens for assay, modality and organism are prepended, and sequences are truncated or padded to N = 1,500. No log transform is mentioned for tokenization — [Nicheformer, Nat Methods 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695652/).
- Shared human–mouse vocabulary of 20,310 gene tokens; pretraining corpus SpatialCorpus-110M — [PMC12695652](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695652/).

**scPRINT (Kalfon et al., Nature Communications 2025)**
- "We directly take the log-transformed counts", using log2(x+1), fed through a two-layer MLP (1→d→d, LayerNorm, ReLU, dropout 0.1). Total transcript count is encoded by a similar MLP on log2(1+t).
- Each gene token is the sum of a frozen ESM2 gene-ID embedding (from the most common protein product), the expression embedding, and a genomic-location positional encoding.
- 2,200 genes per cell: randomly chosen expressed genes, padded with randomly sampled unexpressed genes.
- All of the above — [scPRINT, Nat Commun 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003772/).
- In benchmarking it was run with `how="random expr", max_len=4000` — [Hou et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/). A successor, scPRINT-2 (Dec 2025), changes tokenization and losses and adds an "XPressor" architecture — [scPRINT-2 bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.11.693702.full.pdf).

**CellFM (Zeng et al., Nature Communications 2025; 800M params, 100M human cells)**
- "Normalization and a log1p transformation were employed across all gene expression matrices."
- Continuous value embedding: "We then apply a Multilayer Perceptron (MLP) to map the scalar expression values of genes to embedding vectors", with MLP hidden size 256 and output d = 1536.
- Gene set of 24,078 genes; at most 2,048 genes per cell (random subset of highly expressed genes if more); 20% masked-value MSE pretraining.
- All of the above — [CellFM (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092794/).

**TranscriptFormer (Pearce et al., CZI, bioRxiv Apr 2025)**
- "The model expects unnormalized count data (raw counts)", and "You do not need to perform any additional normalization". Counts are "clipped at 30 by default (as was done in training)", with optional `normalize_to_scale` — [TranscriptFormer GitHub](https://github.com/czi-ai/transcriptformer).
- A generative autoregressive joint model over genes and their counts. It uses ESM-2 protein embeddings so that genes from 12 species share an embedding space — [TranscriptFormer bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.25.650731.full.pdf).
- Variants: TF-Metazoa (112M cells, 12 species, 444M trainable params), TF-Exemplar (110M cells, 5 species), TF-Sapiens (57M human cells, 368M trainable) — [GitHub](https://github.com/czi-ai/transcriptformer).
- Uses "expression-aware multi-head self-attention, causal masking, and a count likelihood" — [Helical model card](https://helical.readthedocs.io/en/latest/model_cards/transcriptformer/).

**State / State Embedding SE-600M (Arc Institute, 2025)**
- "Arc represents each cell as the top 2048 genes ranked by log fold expression level". Gene identity comes from ESM2 embeddings: amino-acid embeddings are mean-pooled into isoform embeddings, and isoforms are mean-pooled into a gene embedding. A projection SiLU(LayerNorm(gW+b)) maps genes to model width. A "soft binning" algorithm plus 2 MLPs produces expression encodings that are added to the gene embeddings. The sequence is [CLS] + 2048 genes + [DS]. Training masks 1,280 genes per cell — [HF Virtual Cell Challenge primer](https://huggingface.co/blog/virtual-cell-challenge).
- The repo's `state emb` preprocessing uses the supplied gene-embedding dict (otherwise one-hot). The normalize + log1p + HVG step described in the README applies to the ST model, not SE — [ArcInstitute/state](https://github.com/ArcInstitute/state). SE-600M is about 0.7B params — [HF SE-600M](https://huggingface.co/arcinstitute/SE-600M).

**SCimilarity (Heimberg et al., Nature 2025) — metric-learning, not a transformer**
- Input is "natural-log normalized per cell with a scaling factor of 10,000" (normalize to 1e4 then log1p) over 28,231 genes. The encoder is an MLP with 1024-1024-1024-128 layers. The model is trained with a triplet loss plus an MSE reconstruction loss — [SCimilarity (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11864978/).

**Other models seen in 2025–2026 benchmarks** (only the extraction details reported by the benchmark):
- CELLama turns the top 30 expressed genes into a text sentence passed through all-MiniLM-L6-v2.
- GenePT-w uses an expression-weighted aggregation of ada-002 text embeddings of genes.
- LangCell, scCello and CellPLM are also included.
- All from [Hou et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/).

**Evidence on encoding choice**
- In a controlled comparison (8 configurations trained from scratch on 10M cells), "hard binning consistently degrades performance across all evaluation metrics" and "soft binning achieves optimal performance with learned embeddings". Learned gene embeddings beat ESM-2 gene embeddings by about 29% relative — [Sadalski, ICLR 2026 Gen² workshop](https://iclr.cc/virtual/2026/10013720).

### Inferences
- **Invariance to the preprocessing CosTaL+ might apply first.** Per-cell rank and per-cell quantile-bin encodings (Geneformer, Nicheformer, scGPT, State's top-2048 ordering) are invariant to multiplying a cell by a constant and to any strictly increasing per-cell transform such as log1p. Within-cell normalize_total and log1p therefore do not change these models' tokens (scGPT's random tie-breaking aside). That is why scGPT's zero-shot path can skip them. Geneformer and Nicheformer are *not* invariant to their **per-gene** corpus scaling (gene nonzero median, or technology-specific nonzero mean), which changes the ranking between genes.
- Library-size and depth information is removed from the input by rank and quantile-bin models. It is explicitly injected by scFoundation (T/S tokens), scPRINT (total-count MLP token) and AIDO.Cell (read-depth-aware objective). TranscriptFormer sees it implicitly through raw counts clipped at 30.
- **Non-determinism (relevant to CosTaL+ reproducibility).** scGPT's official embedding path samples genes when a cell has more than about 1,199 nonzero genes (`sampling=True`, `max_length=1200`) and breaks binning ties at random. scPRINT samples expressed and unexpressed genes. UCE samples genes (not re-verified). Repeated runs without fixed seeds therefore give slightly different embeddings.

### Gaps
- UCE's exact sampling and ordering scheme and CellPLM's input encoding: the UCE preprint and the CellPLM bioRxiv full text could not be retrieved (CellPLM returned HTTP 429). CellPLM treats cells (not genes) as tokens and has spatial positional encoding and a Gaussian-mixture latent prior; this is from memory of the ICLR 2024 paper and not verified.
- The State SE "soft binning" algorithm is not described in the sources I reached; the State preprint was not retrieved.
- AIDO.Cell's exact gene count (likely 19,264, matching scFoundation's list) and its preprocessing (log1p or not) are not stated in the model card.

---

## Q2. What does each model's OUTPUT cell embedding look like (dimension, dense/sparse, sign, L2-normalized or not)?

### Takeaway
Every model outputs a **dense float vector that can take negative values**, sized 128 to 3,072 dimensions. The decisive split for CosTaL+ is normalization. **scGPT (official `embed_data`), UCE and SCimilarity return unit-L2-norm embeddings**, so row norms carry no information and Tanimoto reduces to a monotone function of cosine. **Geneformer, Nicheformer, scFoundation, CellFM, TranscriptFormer and State SE (as far as documented) return un-normalized vectors** (mean-pooled or CLS hidden states). For these, nobody has documented whether the norm is biologically meaningful. One Nicheformer finding shows that token norms can encode technical context such as modality.

### Cited Findings

Summary table. "Unit norm?" means the official extraction code divides by the L2 norm.

| Model | Cell embedding = | Dim (verified where cited) | Unit L2 norm? | Source |
|---|---|---|---|---|
| scGPT | `<cls>` output at position 0 | 512 (whole-human; Census `n_features: 512`) | **Yes** — `cell_embeddings / np.linalg.norm(..., axis=1, keepdims=True)` | [cell_emb.py](https://github.com/bowang-lab/scGPT/blob/main/scgpt/tasks/cell_emb.py); [Census embedding demo](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_embedding.html) |
| UCE | CLS token after a 4-block MLP decoder | 1,280 (`33l_8ep_1024t_1280`) | **Yes** — `nn.functional.normalize(embedding, dim=1)` inside `forward()` | [UCE model.py](https://github.com/snap-stanford/UCE/blob/main/model.py); [Hou et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/) |
| SCimilarity | MLP bottleneck | 128 | **Yes** — outputs "unit length normalized" (on a 128-d hypersphere) | [SCimilarity PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11864978/) |
| Geneformer V1-10M | Mean of non-padding gene embeddings (no CLS in V1) | 256 (hidden_size) | **No** normalization in `emb_extractor.py` | [emb_extractor.py](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/emb_extractor.py); [V1 config](https://huggingface.co/ctheodoris/Geneformer/blob/main/Geneformer-V1-10M/config.json) |
| Geneformer V2-316M (default) | `emb_mode="cls"` by default (V2 has `<cls>`); `"cell"` = mean pooling | 1,152 (hidden_size, 18 layers) | **No** | [emb_extractor.py](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/emb_extractor.py); [V2 config](https://huggingface.co/ctheodoris/Geneformer/blob/main/config.json) |
| scFoundation | `pool_type="all"`: concat of [last token (S/depth), 2nd-to-last token (T), max-pool over genes, mean-pool over genes]; `"max"`: max over all positions | 4·d for "all", d for "max" (d set by checkpoint config; not verified here) | **No** — saved as raw numpy | [get_embedding.py](https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py) |
| Nicheformer | Mean of last-layer gene tokens, **excluding contextual tokens** | 512 (12 layers, 16 heads, 49.3M params) | **No** normalization described | [Nicheformer PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695652/) |
| CellFM | Learnable CLS token | d = 1,536 (hidden width; the cell-embedding dim is not stated explicitly) | Not described | [CellFM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092794/) |
| TranscriptFormer | Mean-pooled across all genes, last layer by default | **Not documented** in README | Not documented | [TranscriptFormer GitHub](https://github.com/czi-ai/transcriptformer) |
| State SE-600M | [CLS] token output | **Not documented** in sources reached | Not documented | [HF VCC primer](https://huggingface.co/blog/virtual-cell-challenge) |
| scPRINT | Several disentangled class-token embeddings (default/general + cell type, disease, sex, ethnicity, assay, organism); classifier logits are also usable | Not stated in visible text (sizes 2M–100M params) | Not described; but a contrastive term uses cosine similarity to push embeddings apart | [scPRINT PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003772/) |
| AIDO.Cell | Not documented in model card | Hidden 128 / 256 / 650 / 1,280 by size | Not documented | [AIDO.Cell-100M](https://huggingface.co/genbio-ai/AIDO.Cell-100M) |
| scBERT | Appended CLS token, final token vector (per benchmark) | dim = 200 | Not documented | [Hou et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/); [scBERT GitHub](https://github.com/TencentAILabHealthcare/scBERT) |

Additional cited details:
- Geneformer `emb_layer` defaults to -1, the "2nd to last layer". The docstring says: "Generally, it is best to extract the 2nd to last layer to get a more general representation." V1 has no `<cls>`, so CLS mode falls back to mean pooling — [emb_extractor.py](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/emb_extractor.py). Benchmarks differ in which mode they use: Hou et al. used `emb_mode="cell"` (mean pooling) for V2-316M ([Hou et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/)), and Kedzierska et al. averaged gene embeddings ([Kedzierska](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007350/)).
- **Norms can carry technical signal.** Nicheformer excludes contextual tokens from pooling "to avoid bias from their high norm". The modality token has a much higher output norm, which biased embeddings toward modality — [Nicheformer PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695652/).
- The SCimilarity triplet loss uses squared Euclidean distance on the unit-normalized embeddings. Its similarity score is "defined as the inverse of the cosine distance of two embedded cell profiles" — [SCimilarity PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11864978/).
- UCE's decoder is `full_block(d_model,1024) → full_block(1024,output_dim) → full_block(output_dim,output_dim) → nn.Linear(output_dim,output_dim)`. Each block is Linear → LayerNorm → GELU → Dropout. Normalization follows the final Linear — [UCE model.py](https://github.com/snap-stanford/UCE/blob/main/model.py).
- scFoundation's "all" pooling concatenates the embeddings of the two read-depth tokens (positions −1 and −2) with the max- and mean-pooled gene embeddings — [get_embedding.py](https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py).

### Inferences
- **Dense, can be negative (all models).** Every output is a final transformer hidden state, a LayerNorm-ed token, a linear-layer output, or an MLP output. No model applies ReLU or softplus at the end, and the documented pipelines apply no sparsification. All embeddings are dense float32 vectors whose coordinates can be negative. This is a structural reading of the code and architectures. One exception: the final activation of SCimilarity's 128-d layer was not checked, but unit normalization does not force positivity either way.
- **Exact algebra for Tanimoto vs cosine.** Write c = cos(a,b) and ρ = ‖a‖/‖b‖. Then T = a·b / (‖a‖² + ‖b‖² − a·b) = c / (ρ + 1/ρ − c).
  - (i) For unit-norm embeddings (scGPT, UCE, SCimilarity), ρ = 1 and T = c/(2 − c). This is strictly increasing in c (dT/dc = 2/(2−c)² > 0). kNN sets and neighbor ranks are identical to cosine's. Tanimoto weights are a fixed convex reshaping of cosine and add **no new information**; they only sharpen the contrast between high and low cosine.
  - (ii) For un-normalized embeddings, the only extra information Tanimoto brings is the **norm ratio** ρ. ρ + 1/ρ ≥ 2, with equality only when the norms are equal, so pairs with mismatched norms are down-weighted.
  - (iii) Because embeddings can be negative, c ∈ [−1, 1] and T ∈ [−1/3, 1]. The denominator is always positive for nonzero vectors: ‖a‖² + ‖b‖² − a·b ≥ (‖a‖² + ‖b‖²)/2. Negative edge weights are possible in principle, but a kNN graph's top-k neighbors will almost always have c > 0.
- **Whether the norm is informative for un-normalized models is unknown, and may be technical rather than biological.** For mean-pooled LayerNorm-ed token outputs (Geneformer "cell" mode, Nicheformer, TranscriptFormer), each token has roughly fixed norm after LayerNorm. The pooled vector's norm then mostly measures how *aligned* the cell's gene tokens are, and number of expressed genes and sequence length can affect it. scFoundation "all" explicitly includes the depth-token embeddings, so its norm plausibly correlates with library size. The Nicheformer modality-norm finding is direct evidence that norms can encode technical context. A CosTaL+ user should check empirically whether row norm correlates with total counts, number of detected genes or batch before relying on Tanimoto.
- **Concatenated or heterogeneous blocks.** scFoundation "all" concatenates four blocks with different scales (max-pool and mean-pool are not comparable). Cosine and Tanimoto on the raw concatenation will be dominated by whichever block has the largest norm. Per-block standardization or L2 normalization is a design choice the tool should flag.

### Gaps
- Cell-embedding dimensions are not documented in the sources I reached for TranscriptFormer, State SE-600M, scPRINT, AIDO.Cell embedding extraction, CellPLM and scFoundation (d not stated). Secondary tutorials (OmicVerse) claim 512 for scPRINT and Nicheformer; only Nicheformer's 512 is confirmed by the primary paper.
- Geneformer V2-104M's hidden size was not fetched (only V1-10M = 256 and V2-316M = 1152 were verified).
- No source found that analyzes or reports the distribution of row norms of foundation-model cell embeddings, or whether the norm correlates with biology versus depth. This is a genuine literature gap.
- Whether CELLxGENE Census-hosted Geneformer/UCE/scGPT embeddings were stored normalized: only the scGPT dimension (512) was visible on the Census page.

---

## Q3. What do independent evaluations say about zero-shot embeddings for clustering and cell-type separation versus HVG+PCA or scVI?

### Takeaway
Independent zero-shot benchmarks from 2025–2026 consistently find that FM cell embeddings do **not** reliably beat simple baselines for unsupervised cell-type clustering. HVG selection and HVG+PCA are often as good or better, and scVI and Harmony beat scGPT and Geneformer on most datasets. The exception is SCimilarity, the metric-learned, unit-normalized model, which edged out PCA in the largest 2026 benchmark. Performance is strongly task- and dataset-dependent, and larger models do not reliably do better.

### Cited Findings
- **Kedzierska et al., Genome Biology 2025** ("Zero-shot evaluation reveals limitations of single-cell foundation models"; preprint 2023)
  - Setup: Geneformer and scGPT (kidney, blood/bone-marrow, and 33M-cell human variants) were used zero-shot, against HVG (2,000 genes), scVI and Harmony, with scVI and Harmony "trained on each individual dataset".
  - Results: "Both models perform worse than selecting highly variable genes (HVG)" on AvgBIO, and "HVG outperforms Geneformer and scGPT across all metrics". scGPT beat scVI and Harmony only on PBMC 12k. "Surprisingly, the best batch integration scores for all datasets were achieved by selecting HVG." "Geneformer consistently ranks last in terms of batch mixing scores."
  - Metrics: AvgBIO is the mean of ASW, NMI and ARI. NMI and ARI were computed on Louvain clusters, at the best of 20 resolutions chosen by NMI.
  - Pretraining objective: without its cell embedding, scGPT "predicts the mean value of the input bin across all binned values".
  - Source: [PMC12007350](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007350/).
- **Hou et al., bioRxiv Jan 2026** ("A unified framework enables accessible deployment and comprehensive benchmarking of single-cell foundation models")
  - 13 scFMs: CELLama, CellFM, CellPLM, Geneformer V2-316M, GenePT-w, LangCell, scBERT, scCello, scFoundation, scGPT, SCimilarity, scPRINT, UCE.
  - Baseline: 2,000 HVG followed by 50-d PCA. Raw counts were the canonical input, and each model ran its own preprocessing.
  - Results: "the classical PCA pipeline achieved similarly strong clustering performance". On the stomach dataset, PCA reached ARI 0.65, exceeded only by SCimilarity at 0.68; PCA ranked second overall. "None of the evaluated models outperformed PCA on spatial transcriptomics data under zero-shot conditions." Increased scale, complexity, corpus size or input encoding does not consistently translate into better downstream performance.
  - Source: [PMC12803055](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/).
- **Liu et al., bioRxiv Jul 2026 (v2)** ("Context-dependent utility and robustness of pretrained single-cell foundation model representations…")
  - Scope: 20 methods, 6 tasks, 1,607 datasets (about 21.8M cells), zero-shot.
  - Results: "scFM performance is strongly task dependent", "no single method consistently outperforming others", and "Conventional statistical and task-specific methods remained competitive in several settings". Greater compute did not consistently mean better performance.
  - Source: [bioRxiv 10.64898/2026.06.18.733285](https://www.biorxiv.org/content/10.64898/2026.06.18.733285).
- **Chen et al., arXiv Jul/Sep 2026** (harmonised benchmark of six models: Nicheformer, CellPLM, scGPT-spatial, GenePT, scELMo, Novae)
  - Covers scRNA-seq, spatial and Perturb-seq data. Expression-trained cell-level transformers best resolved many cell-identity tasks, while "spatial and graph-aware models better preserved tissue architecture". Rankings shifted with modality, preprocessing, tokenisation and metric.
  - Source: [arXiv 2607.17227](https://arxiv.org/abs/2607.17227).
- **scPRINT authors' evaluation:**
  - scPRINT's zero-shot batch correction is "convincing" but "not on par with the SOTA methods scGEN and scVI". scPRINT and scFoundation "show strong zero-shot performances compared to Geneformer v2 and scGPT", which "could not meaningfully correct batch effects without fine-tuning".
  - Source: [scPRINT PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003772/). This is a developer-run comparison.
- **CellFM authors:**
  - "CellFM(800M) obtained low performance on the intra- and inter-datasets when evaluated through zero-shot" (annotation with frozen embeddings plus a classifier). It reports the best scIB AvgBio among FMs, with UCE second.
  - Source: [CellFM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092794/). This is a developer-run comparison with no scVI comparison reported.
- **Nicheformer** is evaluated by linear probing and fine-tuning against Geneformer, scGPT, UCE, CellPLM, scVI and PCA, not by zero-shot clustering — [Nicheformer PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695652/).
- **AIDO.Cell's** abstract claims state-of-the-art zero-shot clustering for the 100M model — [AIDO.Cell bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.28.625303.full.pdf). This is developer-reported, and no independent confirmation was found.

### Inferences
- The best independent zero-shot clustering performer, SCimilarity, is also the one trained explicitly with a metric-learning (triplet) loss on unit-norm embeddings. That suggests embeddings designed for distance comparison suit kNN-graph clustering better than masked-LM by-products. This is a correlation from one benchmark, not a tested causal claim.
- For a graph-clustering tool, HVG+PCA (or scVI) should stay the default input. FM embeddings are an optional alternative input whose value depends on the dataset.

### Gaps
- Boiarsky et al. ("A deep dive into single-cell RNA sequencing foundation models", bioRxiv 2023) was not retrieved. As I recall it (not verified), it showed simple logistic-regression or linear baselines matching fine-tuned scBERT/scGPT on annotation, which is not a zero-shot clustering result.
- No independent zero-shot clustering numbers were found for TranscriptFormer, State SE, AIDO.Cell or CellFM-800M versus scVI or PCA.
- The full method tables of Hou et al. (kNN metric and Leiden resolution of their "unified clustering procedure") are in the truncated part of the paper.

---

## Q4. Are FM embeddings used as input to kNN + Leiden/Louvain clustering in practice, and with what metric?

### Takeaway
Yes. The standard practice in tutorials and benchmarks is to place the embedding in `adata.obsm` and run `scanpy.pp.neighbors(use_rep=...)`, then UMAP and Leiden or Louvain. Usually no metric is specified, so scanpy's default (Euclidean) is used. For unit-norm embeddings this is rank-equivalent to cosine. For un-normalized ones (Geneformer, Nicheformer, scFoundation) it is not, and no source discusses the difference.

### Cited Findings
- The CELLxGENE Census tutorial computes the neighbor graph directly on scGPT embeddings with `scanpy.pp.neighbors(adata, use_rep="scgpt")`, then `scanpy.tl.umap`, without specifying a metric. Hosted embeddings include scVI, Geneformer, scGPT, UCE and NMF (Census 2023-12-15) and are "community-contributed and not actively maintained" — [Census embedding demo](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_embedding.html).
- Kedzierska et al. clustered FM embeddings with Louvain over 20 resolutions; the kNN metric is not specified — [PMC12007350](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007350/).
- Hou et al. clustered all embeddings with a "unified clustering procedure" (details not in the retrieved text). Their kNN-accuracy metric uses Euclidean distance by default, and few-shot prototype inference uses cosine distance — [PMC12803055](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803055/).
- SCimilarity runs kNN search with hnswlib over 23.4M cells (ef_construction=400, M=50, default k=1,000) and on an annotation index of 7.9M cells (k=50). Its similarity score is cosine-based. Hierarchical clustering of cell-type centroids used average linkage with cosine distance. Per-cell annotation needs no clustering ("As each cell is annotated independently, no clustering is required") — [SCimilarity PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11864978/).
- The scGPT reference-mapping tutorial uses faiss for "fast similarity search" on scGPT embeddings. The index type could not be confirmed from the truncated notebook text — [scGPT Tutorial_Reference_Mapping.ipynb](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Reference_Mapping.ipynb).
- Embeddings are stored as `.obsm["X_uce"]` (UCE, [UCE GitHub](https://github.com/snap-stanford/UCE)), `obsm['embeddings']` (TranscriptFormer, [GitHub](https://github.com/czi-ai/transcriptformer)) and `X_state` (State SE, read as `embed_key="X_state"`, [SE600M-embedding README](https://huggingface.co/datasets/VirtualCell2025/SE600M-embedding/blob/main/README.md)). These are the usual hand-off points to `sc.pp.neighbors(use_rep=...)`.

### Inferences
- scanpy's `pp.neighbors` defaults to `metric="euclidean"` (documented scanpy behavior; not re-fetched this session). On unit-norm embeddings (scGPT, UCE, SCimilarity), ‖a−b‖² = 2 − 2·cos, so Euclidean kNN, cosine kNN and Tanimoto-weighted kNN all produce the **same neighbor sets**. CosTaL+'s exact cosine kNN is then interchangeable with the common practice, and its Tanimoto reweighting is a monotone function of cosine. The only effect is how sharply weights fall off before Leiden.
- On un-normalized embeddings (Geneformer, Nicheformer, scFoundation, CellFM, TranscriptFormer, likely State SE), cosine kNN (CosTaL+) and the default Euclidean kNN (common practice) can produce **different** graphs, and Tanimoto adds a norm-ratio penalty. This is the only regime where CosTaL+'s Tanimoto step can change results relative to cosine. Whether that helps depends on whether the norm reflects biology or depth and batch, which is undocumented.
- Practical recommendations for CosTaL+:
  - Detect unit-norm input (all row norms ≈ 1) and warn that Tanimoto ≡ f(cosine).
  - For un-normalized FM embeddings, report the row-norm distribution and its correlation with QC covariates.
  - Never apply sparse or nonnegative assumptions: all FM embeddings are dense and signed.
  - Treat concatenated embeddings such as scFoundation "all" with care, since block scales differ.

### Gaps
- No primary source was found that explicitly recommends cosine versus Euclidean for building kNN graphs on FM embeddings, or that compares Leiden results under the two metrics on FM embeddings.
- The Census "vector search" tutorial (similar-cell search over embeddings) was not retrieved, so its metric is unknown.
