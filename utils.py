# ========================== ENV & ARGS ==========================
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import csv
import json
import math
import string
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from tqdm import trange

warnings.filterwarnings("ignore")

from args import *  # expects get_parser() to be defined in args.py
parser = get_parser()
args = parser.parse_args()

def load_data(dataset: str | None = None):
    """
    Robust TSV loader for train/dev/test.
    - Uses _read_tsv_two_cols (already in utils.py).
    - Train/dev require labels; test is optional and label-free.
    - Accepts an explicit dataset name or falls back to args.dataset.
    Returns: (train_df, dev_df, test_df)
    """
    ds = dataset or args.dataset
    data_dir = Path(getattr(args, "data_dir", "./datasets")).resolve()
    train_path = data_dir / ds / "train.tsv"
    dev_path   = data_dir / ds / "dev.tsv"
    test_path  = data_dir / ds / "test.tsv"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"[load_data] Missing split(s):\n  {train_path}\n  {dev_path}"
        )

    # train/dev must have labels; test may not.
    train_df = _read_tsv_two_cols(str(train_path), require_label=True)
    dev_df   = _read_tsv_two_cols(str(dev_path),   require_label=True)

    if test_path.exists():
        test_df = _read_tsv_two_cols(str(test_path), require_label=False)
    else:
        # empty test if not present
        test_df = pd.DataFrame({"sentence": []})

    # quick sanity prints
    print(f"[load_data] dataset='{ds}'")
    print(f"  train: {len(train_df)} rows | dev: {len(dev_df)} rows | test: {len(test_df)} rows")

    return train_df, dev_df, test_df


# ========================== NUMERIC HELPERS ==========================
def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return Euclidean distances from 1xd vector a to rows of b (n x d)."""
    return np.linalg.norm(b - a, axis=1)


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    """Softmax(x) with overflow protection."""
    m = np.max(x)
    ex = np.exp(x - m)
    s = ex.sum()
    return ex / (s if s != 0.0 else 1.0)


def _entropy_bits(p: np.ndarray) -> float:
    """Entropy (bits): -sum p log2 p."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * (np.log(p) / np.log(2.0))).sum())


def _cluster_entropy_given_centroid(E: np.ndarray, members_idx: List[int], centroid_idx: int) -> float:
    """
    Given member indices and a centroid index (indices w.r.t. E),
    compute p(i) ∝ exp(u(i)), u(i) = distance(centroid, member), then H(p).
    """
    if not members_idx:
        return 0.0
    c = E[centroid_idx]
    M = E[np.array(members_idx, dtype=int)]
    u = _euclidean(c, M)
    p = _softmax_stable(u)
    return _entropy_bits(p)


def _best_centroid_for_cluster(E: np.ndarray, members_idx: List[int], current_centroid_idx: int) -> int:
    """
    Re-choose centroid = argmax_c H(cluster | centroid=c), scanning members.
    """
    if not members_idx:
        return current_centroid_idx
    best_c = current_centroid_idx
    best_H = _cluster_entropy_given_centroid(E, members_idx, current_centroid_idx)
    for cand in members_idx:
        H = _cluster_entropy_given_centroid(E, members_idx, cand)
        if H > best_H + 1e-12:
            best_H, best_c = H, cand
    return best_c


def _entropy_hard_clustering(
    *,
    E: np.ndarray,
    idx2word: np.ndarray,
    word2idx: Dict[str, int],
    centers_tokens: List[str],
    candidate_tokens: List[str],
    num_centroids: int,
    seed: int = 42,
    tie_alpha: float | None = None,   # weight on H_fixed in blend
    tie_eps: float | None = None,     # tie tolerance for H_fixed
) -> Tuple[List[List[int]], List[int]]:
    """
    Fixed-centroid hard-clustering:
      - K fixed centroids chosen from centers_tokens (actual tokens).
      - Each non-centroid token is assigned to exactly one cluster (exclusive membership).
      - Assignment score = H_fixed, with ties resolved by a regularized H_opt.
        score = alpha * H_fixed + (1-alpha) * (reg_hopt * H_opt), on tied clusters only.
      - Centroids remain fixed for probabilities/mapping; we never switch them persistently.
    """
    alpha    = tie_alpha if tie_alpha is not None else float(getattr(args, "cluster_tie_alpha", 0.9))
    eps      = tie_eps   if tie_eps   is not None else float(getattr(args, "cluster_tie_eps", 1e-6))
    reg_hopt = float(getattr(args, "reg_hopt", 1.0))  # 👈 user-defined regularizer on H_opt

    rng = np.random.default_rng(seed)

    center_idx = [word2idx[w] for w in centers_tokens if w in word2idx]
    cand_idx   = [word2idx[w] for w in candidate_tokens if w in word2idx]
    center_idx = np.unique(np.array(center_idx, dtype=int)).tolist()
    cand_idx   = np.unique(np.array(cand_idx, dtype=int)).tolist()

    if len(center_idx) == 0:
        raise ValueError("[entropy_hard] no centers available in embedding vocab.")
    if len(cand_idx) == 0:
        raise ValueError("[entropy_hard] no candidate pool available in embedding vocab.")

    K = int(max(1, min(num_centroids, len(center_idx))))

    # k-means++-style seeding over the candidate center set (keeps real tokens)
    def _kpp_from_indices(E, pool, K, rng):
        first = int(rng.integers(len(pool)))
        C = [pool[first]]
        while len(C) < K:
            # distance to nearest chosen centroid
            d2 = np.full(len(pool), np.inf, dtype=float)
            for i, idx in enumerate(pool):
                for c in C:
                    dist = np.linalg.norm(E[idx] - E[c])
                    if dist < d2[i]:
                        d2[i] = dist
            probs = d2 / (d2.sum() + 1e-12)
            nxt = rng.choice(len(pool), p=probs)
            C.append(pool[int(nxt)])
        return C
    centroids = _kpp_from_indices(E, center_idx, K, rng)          # fixed
    clusters  = [[c] for c in centroids]  # seed with centroid itself
    assigned  = set(centroids)

    C = E[np.array(centroids, dtype=int)]

    # First pass: give each centroid its nearest *distinct* token (if available)
    for j, c_idx in enumerate(centroids):
        remaining = [i for i in cand_idx if i not in assigned and i != c_idx]
        if not remaining:
            continue
        c = E[c_idx]
        R = E[np.array(remaining, dtype=int)]
        d = np.linalg.norm(R - c, axis=1)
        nn = remaining[int(np.argmin(d))]
        clusters[j].append(nn)
        assigned.add(nn)

    # ---------- Greedy assignment using size-normalized ΔH; tie-break with regularized H_opt ----------
    # Assign each remaining token exactly once (hard clustering)
    queue = [i for i in cand_idx if i not in assigned]
    c_pen = float(getattr(args, "dist_entropy_c", 0.5))  # <-- new weight

    for t_idx in queue:
        # 1) distance from token to each fixed centroid
        dists = np.linalg.norm(C - E[t_idx], axis=1)  # shape: (K,)

        # 2) H_fixed if t were added to each cluster
        Hf = np.array([
            _cluster_entropy_given_centroid(E, clusters[j] + [t_idx], centroids[j])
            for j in range(len(centroids))
        ], dtype=float)

        # 3) Base score: maximize -(d + c*H_fixed)  (equivalently: minimize d + c*H_fixed)
        base = -(dists + c_pen * Hf)
        best_base = float(base.max())
        cand_js = [j for j, s in enumerate(base) if (best_base - s) <= eps]

        if len(cand_js) == 1:
            j_star = cand_js[0]
        else:
            # Tie-break: keep your existing “virtual re-center” preference
            best_score = -1e30
            j_star = cand_js[0]
            for j in cand_js:
                members_plus = clusters[j] + [t_idx]
                c_opt = _best_centroid_for_cluster(E, members_plus, centroids[j])   # virtual only
                Hopt  = _cluster_entropy_given_centroid(E, members_plus, c_opt)
                # Blend the base score (already includes the penalty) with a small positive preference for Hopt
                score = alpha * base[j] + (1.0 - alpha) * (reg_hopt * Hopt)
                if score > best_score + 1e-12:
                    best_score = score
                    j_star = j

        clusters[j_star].append(t_idx)
        assigned.add(t_idx)

    return clusters, centroids


def _clusters_to_mapping(
    E: np.ndarray,
    idx2word: np.ndarray,
    clusters: List[List[int]],
    centroids: List[int],
) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
    """
    For token t, its replacements are cluster-mates excluding t.
    Probabilities = softmax of distances to the cluster centroid, then renormalize after removing t.
    """
    sim_word_dict: Dict[str, List[str]] = {}
    p_dict: Dict[str, List[float]] = {}

    for members_idx, c_idx in zip(clusters, centroids):
        if len(members_idx) <= 1:
            t = idx2word[members_idx[0]]
            sim_word_dict[t] = []
            p_dict[t] = []
            continue

        c = E[c_idx]
        M_idx = np.array(members_idx, dtype=int)
        M = E[M_idx]
        u = _euclidean(c, M)
        T = float(getattr(args, "softmax_temp", 1.0))
        p_all = _softmax_stable(-u / max(T, 1e-8))
        
        for local_pos, tok_idx in enumerate(M_idx):
            tok = idx2word[tok_idx]
            others = [idx2word[i] for k, i in enumerate(M_idx) if k != local_pos]
            probs  = np.array([p_all[k] for k in range(len(M_idx)) if k != local_pos], dtype=float)
            s = probs.sum()
            probs = (probs / s) if s > 0 else np.full_like(probs, 1.0 / max(1, len(probs)))
            sim_word_dict[tok] = others
            p_dict[tok] = probs.tolist()

    return sim_word_dict, p_dict


# ========================== DATA HELPERS ==========================
def _read_tsv_two_cols(path: str, want_text: str = "sentence", want_label: str = "label", require_label: bool = True) -> pd.DataFrame:
    """
    Minimal robust TSV reader:
      * Try headered read; if needed, fall back to header=None with provided names.
      * Handle encoding variants.
      * Ensure text column is str; label coerced to int (if required).
    Returns: DataFrame with columns ['sentence'] and (if required) ['label'].
    """
    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    # Try headered read
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, engine="python",
                             quoting=csv.QUOTE_NONE, on_bad_lines="skip")
            if want_text in df.columns and (want_label in df.columns or not require_label):
                out = df[[want_text] + ([want_label] if require_label else [])].copy()
                if require_label:
                    out[want_label] = pd.to_numeric(out[want_label], errors="coerce")
                    out = out.dropna(subset=[want_label])
                    out[want_label] = out[want_label].astype(int)
                out[want_text] = out[want_text].astype(str).str.strip()
                out.rename(columns={want_text: "sentence", want_label: "label"}, inplace=True, errors="ignore")
                return out
        except Exception:
            pass

    # Fallback: assume no header
    for enc in encodings:
        try:
            names = [want_text, want_label] if require_label else [want_text]
            df = pd.read_csv(path, sep="\t", header=None, names=names,
                             encoding=enc, engine="python",
                             quoting=csv.QUOTE_NONE, on_bad_lines="skip")
            if require_label:
                df[want_label] = pd.to_numeric(df[want_label], errors="coerce")
                df = df.dropna(subset=[want_label])
                df[want_label] = df[want_label].astype(int)
            df[want_text] = df[want_text].astype(str).str.strip()
            df.rename(columns={want_text: "sentence", want_label: "label"}, inplace=True, errors="ignore")
            return df[["sentence"] + (["label"] if require_label else [])]
        except Exception:
            pass

    raise RuntimeError(f"Could not read TSV: {path}")


def is_number(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        return False


# ========================== OPTIONAL: DATASET CLASS ==========================
def _get_tokenizer(model_name: str, do_lower_case: bool = True):
    # imported lazily to avoid TF dependency path issues
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, do_lower_case=do_lower_case)


class Bert_dataset(Dataset):
    """Keep only if you still run classification; safe to keep (lazy tokenizer)."""
    def __init__(self, df: pd.DataFrame, tokenizer=None, model_name=None, max_len=None):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len if max_len is not None else args.max_len
        self.tokenizer = tokenizer or _get_tokenizer(model_name or args.model_type, do_lower_case=True)

    def __getitem__(self, index):
        sentence = str(self.df.loc[index, "sentence"])
        enc = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        input_ids      = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(enc.get("token_type_ids", [0] * self.max_len), dtype=torch.long)
        label_val = self.df.loc[index, "label"] if "label" in self.df.columns else 0
        target = torch.tensor(int(label_val), dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, target

    def __len__(self):
        return len(self.df)


# ========================== RUN TAG (EHC) ==========================
def _mapping_tag(args, K: int):
    src  = str(getattr(args, "soft_candidate_source", "dataset")).lower()
    if src not in {"dataset", "embedding", "vocab_file"}: src = "dataset"
    seed = int(getattr(args, "seed", 42))
    return f"eps_{args.eps}_EHC_K{K}_SRC-{src}_seed{seed}"


def _load_embeddings_subset(emb_path, target_words, dtype=np.float32, l2norm=True):
    """
    Stream-load ONLY rows whose token ∈ target_words.
    Returns: E (n x d), idx2word (np.array of str), word2idx (dict[str,int]).
    """
    target = set(target_words)
    E, idx2word = [], []
    # robust open; ignore bad bytes
    with open(emb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            parts = ln.rstrip().split()
            if len(parts) < 2:
                continue
            w, vals = parts[0], parts[1:]
            if w not in target:
                continue
            try:
                vec = np.asarray([float(x) for x in vals], dtype=dtype)
            except ValueError:
                continue
            idx2word.append(w)
            E.append(vec)

    if not E:
        raise RuntimeError("[subset] No embeddings found for dataset tokens. "
                           "Check tokenization/embedding vocab.")

    E = np.asarray(E, dtype=dtype)
    if l2norm:
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    idx2word = np.asarray(idx2word)
    word2idx = {w: i for i, w in enumerate(idx2word.tolist())}
    print(f"[subset] Loaded {E.shape[0]} dataset tokens, dim={E.shape[1]}")
    return E, idx2word, word2idx


# ========================== CORE: BUILD MAPPING ==========================
def get_customized_mapping(eps):
    """
    Entropy-hard clustering ONLY (no top_k).
    Pipeline:
      * centers = tokens from train/dev that exist in embeddings.
      * candidate pool = centers (dataset) or full embedding vocab (embedding), per args.soft_candidate_source.
      * first pass nearest neighbor; then greedy assignment maximizing cluster entropy; centroid switching.
      * mapping: for each token -> cluster-mates excluding self, probs from centroid-distance softmax.
    """
    # ----- 1) Read TSVs -----
    data_dir = Path(getattr(args, "data_dir", "./datasets")).resolve()
    tr_path = data_dir / args.dataset / "train.tsv"
    dv_path = data_dir / args.dataset / "dev.tsv"
    df_train = _read_tsv_two_cols(str(tr_path), require_label=True)
    df_dev   = _read_tsv_two_cols(str(dv_path), require_label=True)

    # debug preview of the token dataset
    print("\n[map] Train head():")
    print(df_train.head(5).to_string(index=False))
    print("[map] Dev head():")
    print(df_dev.head(5).to_string(index=False))

    corpus_train = " ".join(df_train["sentence"])
    corpus_dev   = " ".join(df_dev["sentence"])
    dataset_vocab = set(corpus_train.split()) | set(corpus_dev.split())

    try:
        from nltk.corpus import stopwords
        STOP = set(stopwords.words("english"))
    except Exception:
        STOP = set()
    
    tokens = [t for t in corpus_train.split() if t.isalpha() and t.lower() not in STOP]
    ctr = Counter(tokens)

    min_freq   = int(getattr(args, "min_token_freq", 2))      # keep words seen ≥ 2 times
    max_tokens = int(getattr(args, "max_dataset_tokens", 1200))  # cap vocab size hard
    dataset_vocab = {w for w, f in ctr.most_common(max_tokens) if f >= min_freq}
    print(f"[map] pruned dataset vocab: {len(dataset_vocab)} (min_freq={min_freq}, cap={max_tokens})")

    # ----- 2) Load embeddings -----
    emb_dir  = Path(getattr(args, "embeddings_dir", "./embeddings")).resolve()
    emb_path = emb_dir / f"{args.embedding_type}.txt"
    E, idx2word, word2idx = _load_embeddings_subset(emb_path, dataset_vocab, dtype=np.float32, l2norm=True)

    target_dim = int(getattr(args, "embed_reduced_dim", 32))  # try 32 or even 16
    if E.shape[1] > target_dim:
        d   = E.shape[1]
        rng = np.random.default_rng(int(getattr(args, "seed", 42)))
        R   = rng.standard_normal((d, target_dim)).astype(np.float32) / np.sqrt(d)
        E   = (E @ R).astype(np.float32)
        E   = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        print(f"[map] RP {d}→{target_dim}")
    else:
        E = E.astype(np.float32, copy=False)

    
    print("\n\n----------------------------------------------------- Token data ---------------------------------------------------------\n")
    print(f"\n{E}\n\n")
    print(f"Token Data Shape: {E.shape}")
    


    # ----- 3) Choose centers and candidate pool -----
    centers_tokens = [w for w in dataset_vocab if w in word2idx]
    if not centers_tokens:
        raise RuntimeError("[entropy_hard] no dataset tokens exist in embeddings to seed centroids.")

    candidate_tokens = centers_tokens.copy()
    print(f"[pool] centers={len(centers_tokens)} | candidates={len(candidate_tokens)}")
    
    # ----- 4) Run clustering -----
    N = len(centers_tokens)
    default_K = max(4, min(60, int(np.sqrt(max(1, N)))))   # e.g., ~sqrt(N), capped at 12
    K = int(getattr(args, "num_centroids", default_K))
    print(f"[EHC] Using K={K}")
    clusters, centroids = _entropy_hard_clustering(
                                            E=E,
                                            idx2word=idx2word,
                                            word2idx=word2idx,
                                            centers_tokens=centers_tokens,
                                            candidate_tokens=candidate_tokens,
                                            num_centroids=K,
                                            seed=int(getattr(args, "seed", 42)),
                                            tie_alpha=float(getattr(args, "cluster_tie_alpha", 0.9)),
                                            tie_eps=float(getattr(args, "cluster_tie_eps", 1e-6)),
                                        )
    print(f"[EHC] Built {len(clusters)} clusters (expected K={K}).")

    # ----- 5) Convert to mapping -----
    sim_word_dict, p_dict = _clusters_to_mapping(E=E, idx2word=idx2word,
                                                 clusters=clusters, centroids=centroids)

    # ----- 6) Save artifacts -----
    run_tag = _mapping_tag(args, K)

    # Roots (respect --sim_root and keep p_dict where it is)
    p_dir   = Path("./p_dict") / args.embedding_type / args.mapping_strategy
    sim_dir = Path(getattr(args, "sim_root", "./sim_word_dict")) / args.embedding_type / args.mapping_strategy
    p_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # (A) Save probabilities for runtime reproducibility (unchanged schema)
    with (p_dir / f"{run_tag}.txt").open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(p_dict, ensure_ascii=False, indent=4))

    
    cluster_members = {}
    for members_idx, c_idx in zip(clusters, centroids):
        center_tok  = idx2word[c_idx]
        member_toks = [idx2word[i] for i in members_idx]
        cluster_members[center_tok] = member_toks

    # Write K clusters (exactly K keys)
    with (sim_dir / f"{run_tag}.txt").open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(cluster_members, ensure_ascii=False, indent=4))

    # ----- 7) Stats & optional viz -----
    nonempty = sum(1 for v in sim_word_dict.values() if v)
    avg_sz   = (sum(len(v) for v in sim_word_dict.values()) / max(1, len(sim_word_dict)))
    print(f"\n[entropy_hard] K={K} | centers={len(centers_tokens)} | candidates={len(candidate_tokens)} | "
          f"nonempty={nonempty} | avg_cluster_mates={avg_sz:.2f}", flush=True)

    if getattr(args, "viz_entropy", False):
        try:
            viz_entropy_clusters(args, K)
        except Exception as e:
            print(f"[viz] skipped: {e}", flush=True)

    return sim_word_dict, p_dict


# ========================== VISUALIZATION (2D & 3D) ==========================
def viz_entropy_clusters(args, K):
    """
    Rebuild clusters from sim_word_dict, map to embeddings, color by cluster,
    output: 2D t-SNE/PCA, 3D PCA (two views), histogram, metrics, top clusters, assignments.
    """
    import collections
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    run_tag     = _mapping_tag(args, K)
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    sim_path = Path(getattr(args, "sim_root", "./sim_word_dict")) / args.embedding_type / args.mapping_strategy / f"{run_tag}.txt"
    p_path   = Path("./p_dict") / args.embedding_type / args.mapping_strategy / f"{run_tag}.txt"

    if not sim_path.exists() or not p_path.exists():
        print(f"[viz] mapping artifacts not found for tag={run_tag}\n  {sim_path}\n  {p_path}")
        return

    # with sim_path.open("r", encoding="utf-8") as f:
    #     sim = json.load(f)
    with sim_path.open("r", encoding="utf-8") as f:
        clusters_k = json.load(f)  # centroid -> list[str]


    emb_dir  = Path(getattr(args, "embeddings_dir", "./embeddings")).resolve()
    emb_path = emb_dir / f"{args.embedding_type}.txt"
    if not emb_path.exists():
        raise FileNotFoundError(f"[viz] embeddings not found: {emb_path}")

    # Build the minimal set of tokens we need to draw: mapping keys + their cluster-mates
    needed_words = set()
    for members in clusters_k.values():
        needed_words.update(members)

    # Stream-load just those rows; keeps memory/time small and consistent with K-limited clustering
    E, idx2word, word2idx = _load_embeddings_subset(
                                                        emb_path,
                                                        needed_words,
                                                        dtype=np.float32,
                                                        l2norm=True
                                                    )


    # Dataset vocab for summaries
    data_dir = Path(getattr(args, "data_dir", "./datasets")).resolve()
    tr_path  = data_dir / args.dataset / "train.tsv"
    dv_path  = data_dir / args.dataset / "dev.tsv"
    df_tr = _read_tsv_two_cols(str(tr_path), require_label=True)
    df_dv = _read_tsv_two_cols(str(dv_path),  require_label=True)
    counter = collections.Counter((" ".join(df_tr["sentence"]) + " " + " ".join(df_dv["sentence"])).split())
    vocab = set(counter.keys())

    # Build labels (one label per cluster)
    token_list = []
    label_list = []
    cid = 0
    for _, members in clusters_k.items():
        for w in members:
            token_list.append(w)
            label_list.append(cid)
        cid += 1

    if not token_list:
        print("[viz] nothing to plot (no overlap).")
        return

    keep_indices = []
    labels_clean = []
    for tok, lab in zip(token_list, label_list):
        if tok in word2idx:
            keep_indices.append(word2idx[tok])
            labels_clean.append(lab)

    keep_indices = np.array(keep_indices, dtype=int)
    labels       = np.array(labels_clean, dtype=int)
    E_sel        = E[keep_indices]

    # ADD this line right after E_sel is created:
    base = reports_dir / f"{args.embedding_type}_{args.mapping_strategy}_{run_tag}"

    # Metrics (Euclidean space)
    metrics = {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        if len(np.unique(labels)) >= 2 and len(labels) >= 50:
            metrics["silhouette"]        = float(silhouette_score(E_sel, labels, metric="euclidean"))
            metrics["davies_bouldin"]    = float(davies_bouldin_score(E_sel, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(E_sel, labels))
    except Exception as e:
        print(f"[viz] metrics failed: {e}")

    with (base.parent / f"{base.name}_metrics.json").open("w", encoding="utf-8") as jf:
        json.dump(metrics, jf, ensure_ascii=False, indent=2)
    print("[viz] metrics:", metrics)

    # Histogram of |Y'|
    try:
        member_counts = [len(members) for members in clusters_k.values()]
        plt.figure()
        plt.hist(member_counts, bins=30)
        plt.xlabel("Members per center (|Y'|)")
        plt.ylabel("Centers")
        plt.title("Replacement set size distribution")
        plt.tight_layout()
        plt.savefig(str(base) + "_member_count_hist.png")
        plt.close()
    except Exception as e:
        print(f"[viz] histogram failed: {e}")

    # Dimensionality reduction helper
    def _reduce(E_in, n_components, seed):
        try:
            if n_components == 2:
                perpl = min(30, max(2, len(E_in) // 3))
                X2 = TSNE(n_components=2, random_state=seed, init="random", perplexity=perpl).fit_transform(E_in)
                return X2, "tsne"
            else:
                X3 = PCA(n_components=3).fit_transform(E_in)
                return X3, "pca"
        except Exception as e:
            # fallback to PCA in 2D/3D if TSNE fails
            X = PCA(n_components=n_components).fit_transform(E_in)
            return X, "pca"

    # Sample for plotting speed
    max_points = int(getattr(args, "viz_max_points", 3000))
    rng = np.random.default_rng(getattr(args, "seed", 42))
    if len(E_sel) > max_points:
        sel = rng.choice(len(E_sel), size=max_points, replace=False)
        E_plot, y_plot, idx_plot = E_sel[sel], labels[sel], keep_indices[sel]
    else:
        E_plot, y_plot, idx_plot = E_sel, labels, keep_indices

    # 2D scatter
    try:
        if len(E_plot) >= 5:
            X2, algo2d = _reduce(E_plot, 2, getattr(args, "seed", 42))
            plt.figure()
            plt.scatter(X2[:, 0], X2[:, 1], s=6, c=y_plot)
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.title(f"Entropy-hard clusters (2D {algo2d.upper()})")
            plt.tight_layout()
            plt.savefig(str(base) + (f"_{algo2d}2d.png"))
            plt.close()
        else:
            print("[viz] too few points for 2D scatter; skipping.")
    except Exception as e:
        print(f"[viz] 2D scatter failed: {e}")

    # 3D scatter (two views)
    try:
        if len(E_plot) >= 5:
            X3, _ = _reduce(E_plot, 3, getattr(args, "seed", 42))
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            # View 1
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=6, c=y_plot)
            ax.set_title("Entropy-hard clusters (3D PCA) — view1")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.view_init(elev=20, azim=45)
            fig.tight_layout()
            fig.savefig(str(base) + "_pca3d_view1.png")
            plt.close(fig)

            # View 2
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=6, c=y_plot)
            ax.set_title("Entropy-hard clusters (3D PCA) — view2")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.view_init(elev=15, azim=120)
            fig.tight_layout()
            fig.savefig(str(base) + "_pca3d_view2.png")
            plt.close(fig)
        else:
            print("[viz] too few points for 3D scatter; skipping.")
    except Exception as e:
        print(f"[viz] 3D scatter failed: {e}")

    # Top clusters by corpus frequency
    try:
        by_c = defaultdict(list)
        for idx, c in zip(keep_indices, labels):
            w = idx2word[idx]
            by_c[c].append((w, counter.get(w, 0)))
        top_clusters = sorted(((sum(f for _, f in v), c) for c, v in by_c.items()), reverse=True)[:10]
        with (base.parent / f"{base.name}_top_clusters.txt").open("w", encoding="utf-8") as fh:
            for _, c in top_clusters:
                items = sorted(by_c[c], key=lambda x: (-x[1], x[0]))[:20]
                fh.write(f"[Cluster {c}] " + ", ".join(f"{w}({f})" for w, f in items) + "\n")
        print("[viz] wrote:", str(base) + "_top_clusters.txt")
    except Exception as e:
        print(f"[viz] summary failed: {e}")

    # Assignments CSV (token → cluster_id)
    try:
        rows = [{"token": idx2word[i], "cluster_id": int(c), "freq_in_corpus": int(counter.get(idx2word[i], 0))}
                for i, c in zip(keep_indices, labels)]
        pd.DataFrame(rows).to_csv(str(base) + "_assignments.csv", index=False)
    except Exception as e:
        print(f"[viz] assignments export failed: {e}")


# ========================== TEXT PRIVATIZATION (S1) ==========================
def generate_new_sents_s1(
    df: pd.DataFrame,
    sim_word_dict: Dict[str, List[str]],
    p_dict: Dict[str, List[float]],
    save_stop_words: bool,
    type: str = "train",
    ensure_one_replacement: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Replace tokens using sim_word_dict / p_dict.
    - If `save_stop_words` is True, stopwords are preserved.
    - Numbers are jittered slightly to reduce re-identification.
    - If `ensure_one_replacement` is True, we force at least one semantic-preserving
      replacement per sentence (when possible).
    Writes TSV to:
      ./privatized_dataset/{embedding_type}/{mapping_strategy}/
          eps_{eps}_{privatization_strategy}_save_stop_words_{save_stop_words}/(train|test).tsv
    """
    # ---- stopwords (robust load with fallback) ----
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        # small fallback set if NLTK isn't available
        stop_words = {
            "a","an","the","and","or","but","if","while","with","for","at","by","from",
            "to","in","on","of","is","am","are","was","were","be","been","being","as",
            "it","its","this","that","these","those","he","she","they","we","you","i"
        }

    # ---- RNG ----
    rng = np.random.default_rng(seed if seed is not None else getattr(args, "seed", None))

    # ---- helpers ----
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    def _choose(repls: List[str], p: np.ndarray) -> str:
        # robust renorm if p is bad or mismatched
        if p is None or len(p) != len(repls) or not np.all(np.isfinite(p)) or p.sum() <= 0:
            return rng.choice(repls)
        p = p / p.sum()
        return rng.choice(repls, p=p)

    # ---- main loop ----
    cnt = raw_cnt = stop_cnt = 0
    dataset = df["sentence"].astype(str)
    new_dataset: List[str] = []

    for i in trange(len(dataset)):
        tokens = dataset.iat[i].split()
        out = []
        replaced_flag = False

        # first pass: natural replacements
        for w in tokens:
            if (save_stop_words and w in stop_words) or (w not in sim_word_dict):
                # keep stopwords; jitter numbers a bit
                if w in stop_words:
                    stop_cnt += 1
                    raw_cnt += 1
                if _is_number(w):
                    try:
                        w = str(round(float(w)) + int(rng.integers(1000)))
                    except Exception:
                        pass
                out.append(w)
            else:
                repls = sim_word_dict.get(w, [])
                if len(repls) == 0:
                    out.append(w)
                    raw_cnt += 1
                else:
                    p = np.asarray(p_dict.get(w, []), dtype=float) if w in p_dict else None
                    new_w = _choose(repls, p)
                    out.append(new_w)
                    if new_w == w:
                        raw_cnt += 1
                    else:
                        replaced_flag = True
            cnt += 1

        # second pass (optional): enforce at least one replacement
        if ensure_one_replacement and not replaced_flag:
            # try to flip exactly one eligible token
            for j, w in enumerate(tokens):
                if (save_stop_words and w in stop_words) or (w not in sim_word_dict):
                    continue
                repls = sim_word_dict.get(w, [])
                if len(repls) == 0:
                    continue
                p = np.asarray(p_dict.get(w, []), dtype=float) if w in p_dict else None
                cand = _choose(repls, p)
                if cand != w:
                    out[j] = cand
                    # Adjust counters: we previously counted it as "raw kept"
                    # only if cand==w. Since we changed it now, ensure we
                    # don't overcount raw_cnt. (No need to decrement because
                    # we only incremented raw_cnt when cand==w above.)
                    replaced_flag = True
                    break
            # If still nothing changed, we just leave the sentence as-is.

        new_dataset.append(" ".join(out))

    # ---- write output ----
    out_df = df.copy()
    out_df["sentence"] = new_dataset

    base_dir = Path("./privatized_dataset") / \
               str(getattr(args, "embedding_type", "glove")) / \
               str(getattr(args, "mapping_strategy", "entropy_hard"))
    leaf = f"eps_{getattr(args, 'eps', 1.0)}_" \
           f"{getattr(args, 'privatization_strategy', 's1')}_save_stop_words_{save_stop_words}"
    out_dir = base_dir / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / ("train.tsv" if type == "train" else "test.tsv")
    # TSV without header to match your reader
    out_df.to_csv(out_path, sep="\t", index=False, header=False)

    print(f"[s1] wrote: {out_path} | tokens seen={cnt} | raw_kept={raw_cnt} | stop_kept={stop_cnt}")
    return out_df