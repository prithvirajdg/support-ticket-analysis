# Support Ticket Clustering Project (GCP / Vertex AI Version)

## Goal
Surface specific, actionable user problems from 20K-500K support transcripts by:
1. Disaggregating transcripts into structured summaries
2. Clustering similar problems by meaning (not words)
3. Manually reviewing clusters and assigning names

## Architecture

```
INPUT: CSV with support transcripts (must have 'body' column)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DISAGGREGATION (disaggregate.py)                  │
│  - Tool: Gemini 2.0 Flash via Vertex AI                    │
│  - JSON mode (response_mime_type) for reliable output      │
│  - Concurrent API calls via ThreadPoolExecutor             │
│  - Input: Raw transcript                                   │
│  - Output: 7 fields (see Output Format below)              │
│  - Files: system_prompt.txt, examples.txt                  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: EMBEDDING (embed.py)                              │
│  - Tool: sentence-transformers (all-MiniLM-L6-v2)          │
│  - Runs locally on VM                                      │
│  - Input: summarised_user_problem column                   │
│  - Output: Vector per row (384 dimensions)                 │
│  - Clusters by MEANING, not vocabulary                     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: CLUSTERING (cluster.py)                           │
│  - Tool: UMAP (dimension reduction) + HDBSCAN (clustering) │
│  - Runs locally on VM                                      │
│  - No need to specify number of clusters                   │
│  - Outliers marked as noise (-1)                           │
│  - Tunable via --min-cluster-size                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: MANUAL REVIEW                                     │
│  - User reviews each cluster                               │
│  - Assigns problem name, tags, priority                    │
│  - Output: Final labeled clusters for stakeholders         │
└─────────────────────────────────────────────────────────────┘
```

## Infrastructure

### Where things run

| Component | Runs On | Why |
|-----------|---------|-----|
| disaggregate.py | VM → calls Vertex AI | Gemini is hosted on Vertex AI |
| embed.py | VM (local compute) | sentence-transformers is a local model |
| cluster.py | VM (local compute) | UMAP/HDBSCAN are algorithms, not APIs |

### VM Sizing

| Scale | Recommended VM | Cost/hour |
|-------|----------------|-----------|
| 20K transcripts | e2-standard-4 (4 vCPU, 16GB) | ~$0.13 |
| 100K transcripts | e2-standard-8 (8 vCPU, 32GB) | ~$0.27 |
| 500K transcripts | e2-highmem-8 (8 vCPU, 64GB) | ~$0.45 |

### Cost Estimates (Gemini 2.0 Flash)

| Scale | Gemini API | VM | Total |
|-------|------------|-----|-------|
| 20K | ~$10 | ~$1 | ~$11 |
| 100K | ~$50 | ~$5 | ~$55 |
| 500K | ~$250 | ~$20 | ~$270 |

## Output Format (7 Fields)

Gemini returns (via JSON mode):
```json
{
  "summarised_user_problem": "core problem for clustering",
  "surface_symptom": "what user literally reported",
  "user_goal": "what they were trying to accomplish",
  "impact_type": "blocked|degraded|information|insufficient information",
  "domain": "connectivity|data storage|software|hardware|insufficient information",
  "product_type": "database access|saas platform|website|...",
  "outage_scope": "all users|not applicable|insufficient information"
}
```

### Field Details

| Field | Purpose | Values |
|-------|---------|--------|
| summarised_user_problem | Core problem (used for clustering) | Free text |
| surface_symptom | What user literally reported | Free text |
| user_goal | What they were trying to accomplish | Free text |
| impact_type | Severity | blocked, degraded, information, insufficient information |
| domain | Category | connectivity, data storage, software, hardware, insufficient information |
| product_type | What product is involved | database access, database retrieval, saas platform, website, marketing, portfolio analysis, monitor, insufficient information |
| outage_scope | Who is affected | all users, not applicable, insufficient information |

## Files

```
support-clustering-scale-copy/
├── run.sh                      # Run full pipeline with one command (VM/local)
├── run_workbench.ipynb         # Run full pipeline on Vertex AI Workbench
├── disaggregate.py             # Step 1: Gemini via Vertex AI
├── embed.py                    # Step 2: Embedding (local)
├── cluster.py                  # Step 3: UMAP + HDBSCAN (local)
├── pipeline_utils.py           # Shared utilities
├── system_prompt.txt           # LLM instructions (7 output fields)
├── examples.txt                # Few-shot examples
├── test_clustering.py          # Test script
├── PROJECT_CONTEXT_SUPPORT_CLUSTERING.md
└── data/
    ├── *.csv                   # Input files
    ├── disaggregated.parquet   # After step 1
    ├── embedded.parquet        # After step 2
    └── clustered.parquet       # Final output
```

## Usage

### Prerequisites

1. **GCP Project** with Vertex AI enabled
2. **Authentication**: `gcloud auth application-default login`
3. **Python packages**:
   ```bash
   pip install google-cloud-aiplatform pandas sentence-transformers umap-learn hdbscan pyarrow
   ```

### Run full pipeline

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

### Or run steps individually

```bash
# Step 1: Disaggregate (calls Vertex AI)
python3 disaggregate.py \
    --input "data/*.csv" \
    --output data/disaggregated.parquet \
    --project your-project-id \
    --region us-central1 \
    --workers 10

# Step 2: Embed (runs locally)
python3 embed.py data/disaggregated.parquet data/embedded.parquet

# Step 3: Cluster (runs locally)
python3 cluster.py data/embedded.parquet data/clustered.parquet --min-cluster-size 15
```

### Configuration

Edit `run.sh` to change:
- `GOOGLE_CLOUD_PROJECT` - your GCP project ID
- `REGION` - Vertex AI region (us-central1 recommended, widely available)
- `INPUT_PATTERN` - path to raw transcripts
- `LIMIT` - number of transcripts to process (0 = all)
- `WORKERS` - concurrent API calls (10-20 recommended)
- `MIN_CLUSTER_SIZE` - minimum items to form a cluster

### Run on Vertex AI Workbench

Use `run_workbench.ipynb` to run the entire pipeline on a managed Vertex AI Workbench instance. Everything runs from within the notebook — no terminal or command line needed.

**Data stays within GCP** — the input CSV is pulled from a GCS bucket and results are pushed back to GCS. Raw transcripts never leave the cloud.

#### Setup

1. **Create a Workbench instance** in the GCP Console:
   - Go to **Vertex AI → Workbench → Create New**
   - Pick a machine type matching your scale (see VM Sizing table above)
   - Select any region (Gemini is widely available)

2. **Upload your input CSV to a GCS bucket** (raw data never leaves GCP):
   - **GCP Console:** Go to Cloud Storage → Buckets → open your bucket → create an `inputs/` folder → Upload Files
   - **From a GCP VM or Cloud Shell:** `gsutil cp /path/to/tickets.csv gs://your-bucket/inputs/tickets.csv`
   - **From another GCS location:** `gsutil cp gs://source-bucket/tickets.csv gs://your-bucket/inputs/tickets.csv`

3. **Open JupyterLab** from the Workbench instance

4. **Upload code files** using the JupyterLab file browser (upload button in left sidebar):
   - `run_workbench.ipynb`
   - `disaggregate.py`, `embed.py`, `cluster.py`, `pipeline_utils.py`
   - `system_prompt.txt`, `examples.txt`

   These are small, non-sensitive code files. The input CSV is NOT uploaded here — it comes from GCS.

#### Configuration

Edit the first code cell in the notebook:

```python
PROJECT_ID = "your-actual-project-id"
BUCKET = "your-bucket-name"
GCS_INPUT = "gs://your-bucket-name/inputs/tickets.csv"
LIMIT = 5000                          # start small to validate, set 0 for all
WORKERS = 20                          # concurrent Gemini API calls
MIN_CLUSTER_SIZE = 15
REGION = "us-central1"
```

#### Running the notebook

Run cells in order:

| Cell | What it does | Time (5K rows) |
|------|-------------|-----------------|
| Install deps | `pip install google-cloud-aiplatform sentence-transformers umap-learn hdbscan pyarrow` | ~2 min |
| Pull + verify | Pulls CSV from GCS to VM, verifies all files present | ~1 min |
| Disaggregate | Calls Gemini 2.0 Flash via Vertex AI for each transcript | ~15-30 min |
| Embed | Runs sentence-transformers locally on the VM | ~10 min |
| Cluster | Runs UMAP + HDBSCAN locally on the VM | ~5 min |
| Preview | Shows cluster counts, sizes, and sample problems | instant |
| Push to GCS | Uploads results back to `gs://your-bucket/outputs/` | ~1 min |

#### Authentication

No manual auth setup needed — Workbench VMs automatically use the instance's **default service account**, which already has access to Vertex AI and GCS buckets in the same project. No `gcloud auth` step required.

**Required IAM roles** for the Workbench service account:
- `Vertex AI User` (for Gemini API calls)
- `Storage Object Admin` (for GCS read/write)
- `Service Usage Consumer` (for API access)

#### Checkpoint/resume

If disaggregation fails partway through (API error, timeout, etc.), it saves a checkpoint after every batch of 50 rows. The notebook has a dedicated "Resume from Checkpoint" cell — just run it to pick up where it left off.

#### Long-running jobs (disconnect-safe)

If the job will take >30 minutes (e.g. processing 20K+ transcripts), the browser-based notebook may disconnect. The notebook includes terminal instructions using `nohup` for disconnect-safe execution — see the "Alternative: Run in Terminal" section in the notebook.

#### Stop Workbench when done

Go to **GCP Console → Vertex AI → Workbench** and click **STOP** on your instance. A running Workbench charges ~$8/day even when idle.

## Key Decisions

### Model choice
- Using Gemini 2.0 Flash (cheap, fast, first-party on GCP)
- Flash pricing: $0.10/M input, $0.40/M output
- JSON mode (`response_mime_type`) ensures reliable structured output

### Why Vertex AI?
- Org security requirements (billing through GCP)
- Gemini is first-party on Vertex AI (no third-party quota gates)
- Default service account auth on Workbench — no manual setup

### Why HDBSCAN over KMeans?
- Discovers natural clusters (don't need to guess K)
- Marks outliers as noise (garbage doesn't pollute clusters)
- Variable cluster sizes (real problems aren't evenly distributed)
- Tunable via min_cluster_size

### Why UMAP before HDBSCAN?
- Reduces 384 dimensions to 15
- Makes clustering faster and more accurate
- Preserves semantic relationships

## Scaling Notes

### 20K (current target)
- Works on small VM
- ~1-2 hours total
- ~$11 cost

### 500K (future option)
- Same code, just takes longer
- ~8-12 hours total
- ~$270 cost
- Dev could optimize: more workers, batch processing, resume from failures

## Changes

### Claude Haiku → Gemini 2.0 Flash (Feb 2025)

**Why:** Claude Haiku on Vertex AI required third-party model quota approval (`online_prediction_requests_per_base_model`). New GCP projects start with zero quota for third-party models, and even a single API call was rejected with 429 errors. Quota increase requests require manual Google review and can take hours to days.

**What changed:**
- `disaggregate.py`: Replaced `anthropic.AnthropicVertex` SDK with `vertexai.GenerativeModel` SDK
- Model: `claude-3-haiku@20240307` → `gemini-2.0-flash-001`
- Region: `us-east5` (Claude-only) → `us-central1` (widely available)
- JSON output: Replaced free-form text parsing with `response_mime_type="application/json"` (fewer parse errors)
- Removed prompt caching logic (Gemini is already cheap without it)
- Dependency: `anthropic[vertex]` → `google-cloud-aiplatform` (often pre-installed on Workbench)

**What didn't change:**
- `system_prompt.txt`, `examples.txt` — model-agnostic, work with both
- `embed.py`, `cluster.py`, `pipeline_utils.py` — no LLM dependency
- Pipeline structure, checkpoint/resume, concurrent workers — all identical
- Output schema (7 fields) — same format regardless of model

**Cost impact:** Roughly the same. Gemini 2.0 Flash ($0.10/M input, $0.40/M output) vs Haiku with caching ($0.03/M cached input, $1.25/M output). At scale both land around $10 for 20K transcripts.

**Operational impact:** Gemini works immediately on any GCP project with Vertex AI enabled. No Model Garden opt-in, no third-party quota requests, no waiting for approval.
