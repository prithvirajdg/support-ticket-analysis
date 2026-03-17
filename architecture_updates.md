# Architecture Updates

This document covers planned improvements to the pipeline across three areas: new input sources, scale, and quality assurance.

---

## 1. New Input Source: Screen Recordings

Screen recordings are a higher-fidelity input than text reviews — they capture the exact UI state, error messages, and user intent simultaneously. The integration point is a new pre-step before `disaggregate.py`.

### Pipeline change

```
[video files / GCS URIs]
         ↓
  transcribe_recording.py   ← NEW
         ↓
  [English narrative JSON]  ← same format as batch_summarize.py output
         ↓
  disaggregate.py  →  embed.py  →  cluster.py   (unchanged)
```

### How `transcribe_recording.py` works

- **Input**: CSV of `{recording_id, gcs_uri, metadata...}` or a directory of video files
- **Model**: Gemini 2.0 Flash (multimodal, video input natively supported — already used in the pipeline)
- **System prompt** instructs the model to:
  - Describe what the user was trying to do (intent)
  - Identify the failure moment precisely — error message text, frozen UI, wrong value shown on screen
  - Name specific UI elements, flow steps, and error codes visible in the recording
  - Split into multiple problems if the recording shows more than one distinct failure
- **Output**: same JSON schema as `batch_summarize.py` — `[{narrative, emotion, sentiment}, ...]`
- **Checkpoint/resume**: same pattern as the rest of the pipeline

Screen recordings give higher `cause_fidelity` scores than text reviews because the exact failure state is visible — error codes, wrong fare displayed, stuck loading screens are captured directly rather than described from memory.

---

## 2. Scale

### Current state

| Component | Works today at | Bottleneck |
|---|---|---|
| `disaggregate.py` | ~20K rows | Checkpoint memory, real-time API throughput |
| `embed.py` | ~100K rows | Peak memory (~2.5GB at 500K) |
| `cluster.py` | ~20K rows (global) | UMAP k-NN graph at 500K × 384 dims |

### What breaks and why

**`disaggregate.py` — two issues:**

*Checkpoint is a memory bomb.* The current code accumulates all processed rows as a Python list and serializes the entire list to JSON after every batch of 50:

```python
results = []  # grows unboundedly
save_checkpoint(..., results, ...)  # serializes everything, every 50 rows
```

By the time 200K rows are processed (with ~1.5× multi-problem expansion, ~300K dicts), each checkpoint write takes minutes and produces a file hundreds of MB in size. Fix: write each batch's results directly to Parquet incrementally; checkpoint only stores `processed_rows` + token stats.

*Real-time API throughput ceiling.* At Vertex AI's standard quota for Gemini Flash (~300 RPM), 500K rows takes ~28 hours through a single Python process. Even at 1000 RPM it's ~8 hours, fragile, and burns through quota. Fix: switch to Gemini Batch Prediction API (JSONL → async job → JSONL out), identical in pattern to how `batch_summarize.py` already works with Anthropic's Batch API.

**`embed.py` — minor.**
`model.encode()` returns a 500K × 384 numpy array (~750MB), then `pd.DataFrame()` and `pd.concat()` each copy it. Peak memory is ~2.5GB for embeddings alone. Fine on e2-standard-8 (32GB); needs a VM size recommendation for large runs. The `df.apply(axis=1)` text-building loop is also slow at scale — easy vectorised fix.

**`cluster.py` — mostly fine, needs the right approach.**
The code already uses the `hdbscan` library (not sklearn) with `core_dist_n_jobs=-1`, and UMAP with `low_memory=True` and `n_jobs=-1`. The concern at 500K is global UMAP building a k-NN graph over 500K × 384 dims — roughly 45–90 minutes and 16–32GB RAM on a CPU machine.

The fix is already in the codebase: **use `--stratify-by` for large runs.** With stratification by `team` × `failure_mode` (40 strata at ~12K rows each), each UMAP+HDBSCAN call is over a small, semantically homogeneous subset. Stratification should be the recommended default for runs >50K rows.

### Target scale and what's needed

**100–200K rows:**
- Fix checkpoint architecture in `disaggregate.py` (incremental writes)
- Add rate limiting + exponential backoff
- Request quota increase to 1000–2000 RPM on the GCP project
- Run cluster with `--stratify-by team failure_mode`
- No architecture change needed; a 200K run would complete in 3–5 hours

**500K rows:**
- All of the above, plus:
- Rewrite `disaggregate.py` to use Gemini Batch Prediction API
- Clustering already tractable with stratification on a e2-highmem-8 (64GB)

---

## 3. Quality Checks

Two new quality checks are needed — one after disaggregation, one after clustering. Both are **read-only analysis** (no data is modified). They would live in a new `validate.py` script.

### 3.1 Disaggregation quality

**Field enum validation — did the LLM stay in-schema?**

Every value for `journey`, `team`, `fidelity`, `failure_mode`, and `impact` is checked against the allowed lists from the schema. Any value not in the list is a hallucination or schema drift. Output: count of violations per field + example values.

**Rate and distribution anomalies:**

| Metric | Flag threshold | What it means |
|---|---|---|
| Parse error rate (PARSE_ERROR + API_ERROR) | > 3% | Prompt/quota problem |
| Fallback rate (insufficient/no actionable info) | > 50% | Input quality low, or prompt too strict |
| Multi-problem explosion factor (output rows / input rows) | > 3× | Splitting logic too aggressive |
| Field concentration (single value > 80% of rows) | > 80% in one journey or team | Surfaced as a warning — may be real signal |

**Problem text sanity:**

- `summarised_problem` shorter than 5 words — LLM produced a placeholder
- `summarised_problem` containing "unknown", "N/A", or "error" — error sentinel leaked into the main output field

### 3.2 Clustering quality

**Structural health:**

| Metric | Flag threshold | What it means |
|---|---|---|
| Noise rate (cluster = -1) | > 40% | `min_cluster_size` too large, or data too sparse |
| Cluster count | < 10 for >10K rows | Over-merged; parameters need tuning |
| p10 cluster size | < 5 | Hundreds of micro-clusters that are unusable |

**Per-cluster purity:**

For each cluster, compute the fraction of rows sharing the dominant value for `journey`, `team`, and `failure_mode`. A cluster that is 50/50 split on `journey` is a mixed cluster and should be flagged. The existing `compare_clusters.py` computes mean purity across all clusters — this extends it to per-cluster output so mixed clusters are individually identifiable.

Flag any cluster where the top-2 values for any structural field are within 10 percentage points of each other.

**Coherence spot-check:**

For each cluster, surface the 3 problems with highest cosine distance from the cluster centroid. If they are clearly different topics, the cluster is too broad. This is a human review aid, not an automated pass/fail.

---

## 4. Full Change List

| # | Change | File(s) | Complexity | Risk |
|---|---|---|---|---|
| 1 | Fix checkpoint — incremental Parquet writes, don't accumulate in memory | `disaggregate.py` | Small (2–3 hrs) | **Medium** — changes how output is written; if wrong, partial results could be lost or duplicated on resume |
| 2 | Rate limiting + exponential backoff on Vertex AI calls | `disaggregate.py` | Small (1–2 hrs) | Low — additive, only triggers on quota errors |
| 3 | Schema enum validation — flag out-of-schema field values post-disaggregation | `disaggregate.py` | Small (1–2 hrs) | Very low — additive, non-blocking |
| 4 | Disaggregation quality report (error rates, distributions, text sanity) | new `validate.py` | Small (3–4 hrs) | Zero — read-only analysis |
| 5 | Clustering quality report (noise rate, per-cluster purity, size distribution) | `validate.py` + `compare_clusters.py` | Small (3–4 hrs) | Zero — read-only analysis |
| 6 | Rewrite `disaggregate.py` to use Gemini Batch Prediction API | `disaggregate.py` | Large (1 day) | **High** — complete rewrite of main processing loop; different API surface; GCS dependency; must be tested end-to-end before use on real data |
| 7 | Fix `apply(axis=1)` → vectorised text build | `embed.py` | Tiny (30 min) | Very low — performance only, output identical |
| 8 | Add customer-side examples | `examples.txt` | Small (half day) | Low — prompt change; needs a test run to verify output quality doesn't regress on partner-side examples |
| 9 | Build `transcribe_recording.py` | new file | Medium (half day) | Medium — new API surface (Gemini video input), new input format; validate against known recordings before use in production |
| 10 | Document VM and quota requirements for 100K and 500K runs | `PROJECT_CONTEXT_SUPPORT_CLUSTERING.md` | Tiny (30 min) | Zero |

### Recommended order

1. **3, 4, 5** first — zero/very low risk, directly improve output confidence
2. **1, 2** next — unlocks 100–200K runs
3. **6** — unlocks 500K; test on a 10K subset before full run
4. **8, 9** are independent and can run in parallel with anything above
5. **7, 10** any time
