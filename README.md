# CusText: Entropy‑Guided Clustering for Semantic Text Privatization

This repository implements a *cluster‑based word replacement* pipeline for privacy‑preserving text augmentation. At a high level:

1. **Load data** (`datasets/<dataset>/{train,dev,test}.tsv`) and read `"sentence"` (and `"label"` where present).
2. **Build a vocabulary** from the dataset and **load embeddings** for those tokens from `embeddings/<embedding_type>.txt` (vectors are L2‑normalized).
3. **Cluster the tokens** into **K semantic clusters** (medoid/centroid based, entropy‑aware assignment).
4. **Derive a token→candidate mapping**: for each token, candidate replacements are its cluster mates; per‑token probabilities come from distances to the cluster centroid (softmax with optional temperature).
5. **Rewrite text**: create privatized `train.tsv` / `test.tsv` by probabilistically replacing tokens, optionally preserving stopwords and jittering numbers.
6. **(Optional) Visualize** clusters: 2D/3D scatter plots + histograms + intrinsic metrics saved under `reports/`.

> **Why “K clusters” and also a “token map”?**  
> The **K clusters** are the *semantic groups* your algorithm builds for the dataset vocabulary. The **token map** is a *per‑token view* that lists each token’s candidate replacements and probabilities (often much larger than K because it has one entry per token).


---

## Repository layout

- `main.py` — Orchestrates the end‑to‑end run: builds mapping, rewrites datasets, and (optionally) trains a classifier on privatized data.
- `utils.py` — Core implementation: robust TSV reader, embedding subset loader, entropy‑guided clustering, mapping construction, text rewriting, and visualizations.
- `args.py` — CLI flags and defaults (dataset paths, number of clusters, temperatures, output locations, etc.).

---

## Algorithm overview

### 1) Data loading
- `train.tsv` and `dev.tsv` must contain `sentence` and `label`. `test.tsv` may omit `label`.
- A resilient reader tries multiple encodings, accepts header/no‑header formats, coerces the `label` to int, and strips text.

### 2) Embedding subset
- Only tokens present in the dataset vocabulary are streamed from `embeddings/<embedding_type>.txt`. Vectors are L2‑normalized column‑wise to keep distances well‑behaved.

### 3) Entropy‑guided hard clustering
- **Seeding:** Pick `K` centers from dataset tokens (K is controlled by `--num_centroids`).  
- **Assignment rule:** For each remaining token, compute the cluster’s **fixed‑centroid entropy** if the token were added. Assign to the cluster that maximizes this `H_fixed`.  
- **Tie‑breaking:** When multiple clusters are within a small `ε` of the best `H_fixed`, compute a **virtual re‑centering entropy** `H_opt` (best member as centroid) and blend the two:  
  `score = α · H_fixed + (1−α) · (reg_hopt · H_opt)`  
  and choose the cluster with the highest `score`. This stabilizes assignments and combats degenerate local choices.
- **Outputs:** Two artifacts are written:  
  - `sim_word_dict/<embedding_type>/<mapping_strategy>/<tag>.txt` — cluster membership manifest and per‑token candidate lists  
  - `p_dict/<embedding_type>/<mapping_strategy>/<tag>.txt` — per‑token replacement probabilities

### 4) Token→candidate mapping
For each token *t* in a cluster, its candidate set is the other members of the same cluster. The per‑token probability vector is obtained by softmaxing the **(negated) distances to the cluster centroid**. A **temperature** `T` sharpens or smooths the distribution: lower `T` → pick closer words more often; higher `T` → flatter. (If probabilities are degenerate or ill‑conditioned, they are renormalized defensively.)

### 5) Text privatization (strategy `s1`)
- Iterate each sentence; for each token:  
  - keep stopwords if `--save_stop_words` is set;  
  - jitter numeric strings slightly to discourage linkage;  
  - otherwise, sample a replacement from that token’s candidate list using the per‑token probabilities.  
- An optional **“ensure‑one‑replacement”** pass can force at least one semantic‑preserving change per sentence (if applicable).  
- Writes to `privatized_dataset/<embedding_type>/<mapping_strategy>/eps_<eps>_<strategy>_save_stop_words_<flag>/{train|test}.tsv`.

### 6) Visualization & diagnostics (optional)
- Produces intrinsic metrics (Silhouette, DB‑Index, Calinski‑Harabasz), a histogram of candidate‑set sizes, and 2D/3D scatter plots for quick sanity checks.
- Saved under `reports/entropy_kmeans/` (configurable via `--reports_dir`).

---

## Key CLI arguments

> See `args.py` for the full list and defaults.

- **Data & I/O**  
  `--dataset` (default: `sst2`), `--data_dir` (`./datasets`),  
  `--embeddings_dir` (`./embeddings`), `--sim_root` (`./sim_word_dict`),  
  `--reports_dir` (`./reports/entropy_kmeans`)

- **Clustering / mapping**  
  `--num_centroids` (K), `--mapping_strategy` (namespace), `--cluster_prob {uniform,softmax}`,  
  `--softmax_temp` (softmax temperature for replacements), `--eps` (tagging/experiment id)

- **Privatization**  
  `--privatization_strategy s1`, `--save_stop_words {True|False}`

- **Viz**  
  `--viz_entropy` (save plots/metrics), `--viz_max_points` (downsampling), `--annotate_top_n`

- **Repro/other**  
  `--seed`, `--lambda_reg`, `--max_iter`, `--tol`

---

## Typical run

```bash
# 1) Build mapping + write artifacts + visualize clusters
python main.py \
  --dataset sst2 \
  --embedding_type cf-vectors \
  --mapping_strategy conservative \
  --num_centroids 30 \
  --eps 1.0 \
  --viz_entropy

# 2) Privatize splits using strategy s1 (done automatically by main.py when --privatization_strategy s1)
#    Outputs go to: privatized_dataset/<embedding_type>/<mapping_strategy>/...
```

**Inputs expected**
```
datasets/
  sst2/
    train.tsv   # sentence<TAB>label
    dev.tsv     # sentence<TAB>label
    test.tsv    # sentence   (label optional)
embeddings/
  cf-vectors.txt
```

**Outputs produced**
```
sim_word_dict/<embedding_type>/<mapping_strategy>/<tag>.txt   # token→candidate list per cluster mapping
p_dict/<embedding_type>/<mapping_strategy>/<tag>.txt          # token→probabilities
privatized_dataset/<embedding_type>/<mapping_strategy>/...    # new train/test .tsv
reports/entropy_kmeans/...                                    # png/json diagnostics when --viz_entropy
```

---

## Notes & design choices

- **K vs. token map size:** You set K via `--num_centroids`. The cluster **count** is K; the **token map** size equals the number of tokens handled (often >> K).  
- **Numerical stability:** distance→probability uses a stable softmax and renormalization; if a token has no valid candidates, it is left untouched.  
- **Stopword & number handling:** stopwords may be preserved; numbers receive small random offsets.  
- **Determinism:** set `--seed` to make centroid seeding and sampling reproducible.

---

## Troubleshooting

- *“Too many clusters in the dump”*: ensure you’re inspecting the **cluster dump/tag** corresponding to the K‑way run (not the per‑token map, which enumerates every token).  
- *“No plots”*: pass `--viz_entropy` and check `--reports_dir`. If very few points are available, metrics/plots may be skipped by design.  
- *“Replacement probabilities look flat/peaky”*: adjust `--softmax_temp` (lower = peakier; higher = flatter).

---

## Citation / Attribution

This README describes the concrete implementation in this codebase; please cite the repository or accompanying report if you publish results.
