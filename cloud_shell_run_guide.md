# Cloud Shell Run Guide

Run the full disaggregation → embedding → clustering pipeline on up to ~2K reviews
directly in Google Cloud Shell — no Workbench or VM required.

---

## Prerequisites

- Access to Google Cloud Console with Cloud Shell
- Vertex AI API enabled on your GCP project
- IAM role: `roles/aiplatform.user` on the project
- The repo cloned or files uploaded to Cloud Shell

---

## Files required in Cloud Shell

All files must be in the same directory (e.g. `~/poc/`):

```
~/poc/
├── disaggregate.py
├── embed.py
├── cluster.py
├── pipeline_utils.py
├── system_prompt.txt
├── examples.txt
└── data/
    └── porter_partner_5000_summarized - input_set.csv
```

The input file must have a `body` column containing the English review narrative
(output of `batch_summarize.py`). The pre-summarized file above already has this.

---

## Full run — copy and paste in order

### 0. Open Cloud Shell and navigate to the project

```bash
cd ~/poc
```

---

### 1. Install dependencies

```bash
pip install --quiet \
  google-cloud-aiplatform \
  pandas pyarrow \
  sentence-transformers \
  umap-learn hdbscan \
  scikit-learn numpy
```

> Takes ~3-5 minutes. Only needed once per Cloud Shell session (or after session reset).

---

### 2. Confirm GCP project and enable Vertex AI

```bash
# Check current project
gcloud config get-value project

# Set project if wrong
gcloud config set project YOUR_PROJECT_ID

# Enable Vertex AI API if not already enabled
gcloud services enable aiplatform.googleapis.com
```

---

### 3. Create data directory and prepare 2K input slice

```bash
mkdir -p data logs

python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/porter_partner_5000_summarized - input_set.csv")
df.head(2000).to_csv("data/input_2k.csv", index=False)
print(f"Saved {len(df.head(2000))} rows")
print(f"Columns: {df.columns.tolist()}")
EOF
```

---

### 4. Step 1 — Disaggregate (Gemini via Vertex AI)

```bash
python3 disaggregate.py \
  --input data/input_2k.csv \
  --output data/disaggregated.parquet \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --workers 5 \
  --batch-size 50
```

**Expected time:** ~8-12 minutes for 2K rows.

**If the session drops mid-run**, resume from the last checkpoint:

```bash
python3 disaggregate.py \
  --input data/input_2k.csv \
  --output data/disaggregated.parquet \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --workers 5 \
  --batch-size 50 \
  --resume
```

---

### 5. Step 2 — Embed

```bash
python3 embed.py \
  data/disaggregated.parquet \
  data/embedded.parquet
```

**Expected time:** ~1 minute. First run downloads the `all-MiniLM-L6-v2` model (~90MB).

---

### 6. Step 3 — Cluster

```bash
python3 cluster.py \
  data/embedded.parquet \
  data/clustered.csv \
  --min-cluster-size 5 \
  --umap-dims 10 \
  --umap-neighbors 15
```

**Expected time:** ~1-2 minutes.

---

### 7. Inspect results

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/clustered.csv")
n_clusters = len(df[df.cluster >= 0].cluster.unique())
n_noise = (df.cluster == -1).sum()
print(f"Total rows  : {len(df)}")
print(f"Clusters    : {n_clusters}")
print(f"Noise (-1)  : {n_noise} ({100*(df.cluster==-1).mean():.1f}%)")
print()
print(df.groupby('cluster')[['journey', 'team', 'summarised_problem']] \
  .first().sort_values('cluster').to_string())
EOF
```

---

## Parameter reference

### disaggregate.py

| Parameter | Value used | Default | Notes |
|-----------|-----------|---------|-------|
| `--input` | `data/input_2k.csv` | — | Required |
| `--output` | `data/disaggregated.parquet` | — | Required |
| `--project` | your GCP project ID | `$GOOGLE_CLOUD_PROJECT` | Required |
| `--region` | `us-central1` | `us-central1` | Change if your quota is in another region |
| `--workers` | `5` | `10` | Concurrent Gemini API calls; keep ≤5 on Cloud Shell to avoid quota errors |
| `--batch-size` | `50` | `50` | Rows per checkpoint save |
| `--limit` | _(omit = all)_ | `0` (all) | Set to e.g. `100` for a quick smoke test |
| `--resume` | flag | off | Add if resuming after a failure |

### embed.py

| Parameter | Value used | Default | Notes |
|-----------|-----------|---------|-------|
| `input` | `data/disaggregated.parquet` | — | Positional, required |
| `output` | `data/embedded.parquet` | — | Positional, required |
| `--batch-size` | _(omit)_ | `256` | Reduce to `32` if Cloud Shell runs low on memory |

### cluster.py

| Parameter | Value used | Default | Notes |
|-----------|-----------|---------|-------|
| `input` | `data/embedded.parquet` | — | Positional, required |
| `output` | `data/clustered.csv` | — | Positional, required |
| `--min-cluster-size` | `5` | `15` | Minimum reviews to form a cluster; increase for broader clusters |
| `--umap-dims` | `10` | `15` | UMAP output dimensions; keep ≤ n_samples/10 |
| `--umap-neighbors` | `15` | `30` | UMAP neighbourhood size; keep well below row count |
| `--prob-threshold` | _(omit)_ | `0.0` | Set e.g. `0.05` to push borderline points to noise |
| `--stratify-by` | _(omit)_ | off | e.g. `--stratify-by team failure_mode` for stratified clustering |

---

## File flow

```
input_2k.csv
    │
    ▼ disaggregate.py  (Vertex AI Gemini 2.0 Flash)
disaggregated.parquet  (+10 structured fields per row; multi-problem rows exploded)
    │
    ▼ embed.py  (all-MiniLM-L6-v2, local)
embedded.parquet  (+384 embedding columns)
    │
    ▼ cluster.py  (UMAP + HDBSCAN, local)
clustered.csv  (cluster column added; embedding columns removed)
```

Auto-generated during the run:
- `data/disaggregated.checkpoint.json` — deleted on clean completion; used by `--resume`
- `logs/` — one log file per step with timestamps and token usage

---

## Scale limits on Cloud Shell

| Scale | End-to-end | Notes |
|-------|-----------|-------|
| 2K rows | ✅ ~15 min total | Default for POC |
| 20K rows | ✅ ~2-3 hrs | Use tmux; all params same |
| 50K rows | ⚠️ borderline | Reduce `--umap-neighbors 10`; UMAP may OOM |
| 100K+ rows | ❌ | Needs a machine with 8GB+ RAM |

For 20K+ rows, start a tmux session before running to survive browser disconnects:

```bash
tmux new -s pipeline
# run your commands
# Ctrl+B then D to detach
# tmux attach -t pipeline to reattach
```

---

## Troubleshooting

**`Permission denied` on Vertex AI**
→ Check IAM: you need `roles/aiplatform.user` on the project.

**High PARSE_ERROR rate (>20%)**
→ Check that the `body` column contains clean English narrative, not raw multilingual text.

**`n_neighbors` error in cluster.py**
→ Reduce `--umap-neighbors` (must be less than the number of embeddable rows after filtering).

**Out of memory in embed.py**
→ Add `--batch-size 32` to the embed.py command.

**Cloud Shell session expired mid-disaggregation**
→ Rerun with `--resume` — the checkpoint file picks up from the last saved batch.
