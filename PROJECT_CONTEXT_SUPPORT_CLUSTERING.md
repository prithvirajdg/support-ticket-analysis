# Porter Review Clustering + Classification Project

## Goal
Surface specific, actionable user problems from Porter Google Play Store reviews (partner and customer apps) by:
1. Summarising multilingual reviews into English narratives
2. Extracting a structured core problem statement (7 fields) from each narrative
3. Clustering similar problems by meaning (not words)
4. Manually reviewing clusters and assigning names

## Background
This project is a fork of `porter-clustering-tickets`, created to add:
- A richer disaggregation output schema (7 fields instead of 1)
- A structured problem definition rubric (pain-point + specific touchpoint required)
- Journey, team, and problem-type taxonomy from Porter's domain
- A probability threshold lever for cluster quality control
- (Planned) A classification layer for known problem buckets

The original `porter-clustering-tickets` folder is preserved unchanged. This folder (`porter-clustering-classification-tickets`) contains all new work.

## Data Source
Reviews are scraped from the Google Play Store using `porter_reviews.py` (lives in `porter-playstore-reviews/`):
- **Partner app**: `com.theporter.android.driverapp`
- **Customer app**: `com.theporter.android.customerapp`
- Fetches up to 25K reviews per app, sorted newest-first, India region
- Output: timestamped CSVs (e.g. `porter_partner_reviews_20260217.csv`)

## Full Pipeline

```
porter_reviews.py          →  raw CSVs (Play Store reviews)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  PRE-STEP: SUMMARISATION (batch_summarize.py)              │
│  - Lives in porter-playstore-reviews/                      │
│  - Tool: Claude Haiku via Anthropic Batch API              │
│  - Prompt caching on system prompt (cost reduction)        │
│  - Input: raw review text (multilingual)                   │
│  - Output: body = English narrative summary per problem    │
│  - Splits multi-problem reviews into one row per problem   │
│  - Also outputs: sentiment, emotion, problem_number        │
│  - Files: system_prompt_summarize.txt, examples_summarize.txt     │
└─────────────────────────────────────────────────────────────┘
           │  (output CSV has 'body' column)
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DISAGGREGATION (disaggregate.py)                  │
│  - Tool: Gemini 2.0 Flash via Vertex AI                    │
│  - JSON mode (response_mime_type) for reliable output      │
│  - Concurrent API calls via ThreadPoolExecutor             │
│  - Checkpoint/resume capability                            │
│  - Input: body (narrative summary from batch_summarize)    │
│  - Output: 9-field structured JSON per problem             │
│    - summarised_problem, fidelity, journey, stage,         │
│      mechanism, team, problem_type, impact, cause_fidelity │
│  - Multi-problem reviews exploded into one row per problem │
│  - Files: system_prompt.txt, examples.txt                  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: EMBEDDING (embed.py)                              │
│  - Tool: sentence-transformers (all-MiniLM-L6-v2)          │
│  - Runs locally on VM                                      │
│  - Input: structured text combining all 7 fields           │
│  - Output: Vector per row (384 dimensions)                 │
│  - Clusters by MEANING + STRUCTURE, not vocabulary         │
│  - Skips rows with error/fallback values                   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: CLUSTERING (cluster.py)                           │
│  - Tool: UMAP (dimension reduction) + HDBSCAN (clustering) │
│  - Runs locally on VM                                      │
│  - No need to specify number of clusters                   │
│  - Outliers marked as noise (-1)                           │
│  - Tunable via --min-cluster-size and --prob-threshold     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: MANUAL REVIEW                                     │
│  - User reviews each cluster                               │
│  - journey/team/problem_type columns allow quick filtering │
│  - Assigns problem name, tags, priority                    │
│  - Output: Final labeled clusters for stakeholders         │
└─────────────────────────────────────────────────────────────┘
```

## Why Two LLM Steps?

The summarisation step (`batch_summarize.py`) produces a **narrative**: a detailed English description of the problem that preserves emotion, evidence, context, and workarounds. This is useful for human reading but too verbose and varied for clustering.

The disaggregation step (`disaggregate.py`) distills the narrative into a **short, abstract core problem statement** — stripping out emotion, evidence, and specific details — so that semantically similar problems cluster together reliably.

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

## Output Format

### batch_summarize.py output (input to this pipeline)
```json
{
  "sentiment": "positive|negative|mixed|neutral",
  "emotion": "frustrated|angry|satisfied|...",
  "body": "narrative summary of the problem in English",
  "problem_number": 1,
  "total_problems": 2
}
```
Plus all original columns from the scraped review CSV.

### disaggregate.py output (7 fields)
Gemini returns (via JSON mode) either a single object or an array of objects:

**Single problem:**
```json
{
  "summarised_problem": "short core problem statement for clustering",
  "fidelity": "pain_only | pain_and_touchpoint | feature_request",
  "journey": "Order Allocation | Order Execution | Payments | ...",
  "stage": "coordinate loading | document approval | allocation delay | ... | unknown | N/A",
  "mechanism": "waiting time compensation | document verification | allocation delay | trip start button | ... | unknown",
  "team": "LFC | Marketplace | TNS | CGE | HSC | unknown",
  "problem_type": "touchpoint_app | touchpoint_ops | service | policy | service/policy | touchpoint_app/service | touchpoint_ops/policy | unknown",
  "impact": "financial_loss | blocked | inconvenienced | informative | unknown",
  "cause_fidelity": "cause_known | cause_unknown | N/A"
}
```

**2–3 distinct extractable problems (one row per problem after explode):**
```json
[
  {"summarised_problem": "...", "fidelity": "...", "journey": "...", "team": "...", "problem_type": "...", "impact": "...", "cause_fidelity": "..."},
  {"summarised_problem": "...", "fidelity": "...", "journey": "...", "team": "...", "problem_type": "...", "impact": "...", "cause_fidelity": "..."}
]
```

**Fallback values (skipped in embed step):**
- `"summarised_problem": "insufficient information"` — pain present but touchpoint too vague
- `"summarised_problem": "no actionable information"` — frustration but no specific touchpoint
- `"summarised_problem": "multiple problems - to be broken down"` — more than 3 distinct problems or vague laundry list
- `"summarised_problem": "PARSE_ERROR"` — JSON parse failure
- `"summarised_problem": "API_ERROR"` — Vertex AI API failure

The `body` column (narrative) is preserved in the output alongside all 7 extracted fields.

## Problem Definition

There are three valid output types:

**Core problem** — requires both:
1. **A pain-point** — something that hurt, blocked, or frustrated the user
2. **A specific touchpoint, service, or policy** — the specific part of Porter's app, service, or policy that caused or failed to resolve the pain

**Feature request** — a specific, bounded new capability the user is asking for with no existing step failing.

**Fallback** — if neither core problem nor feature request criteria are met.

### Fidelity levels
- `pain_only` — pain-point identifiable, but touchpoint too vague to pin down
- `pain_and_touchpoint` — both pain-point and specific touchpoint explicitly stated or unambiguously implied
- `feature_request` — user asking for something new; no existing step is broken

Only `pain_and_touchpoint` rows have meaningful journey/team/problem_type/cause_fidelity values.

### Impact levels
- `financial_loss` — money directly lost, owed, or at stake
- `blocked` — cannot proceed with work or intended action at all
- `inconvenienced` — can proceed but with significant friction
- `informative` — observed an issue but no direct harm occurred

### Cause fidelity
- `cause_known` — trigger or condition that caused the failure is explicitly stated in the review
- `cause_unknown` — touchpoint and failure mode are identifiable but why it failed is not stated
- `N/A` — for pain_only and feature_request rows

### Journey taxonomy
**Partner side:** Onboarding, Ready to Take Orders, Order Allocation, Order Acceptance, Order Execution, Post Trip, Account & Compliance, Performance, Customer Support, Payments, Profile

**Customer side:** Authentication, Service Selection, Booking Details, Vehicle Selection, Order Review & Placement, Partner Allocation, Pre-Pickup, In-Trip, Trip Completion

**Cross-cutting:** Payments, Ratings, Cancellations

### Team taxonomy
| Team | Owns |
|------|------|
| CGE | Customer onboarding, service selection, vehicle selection, customer location issues, ratings |
| Marketplace | GPS, pricing, order allocation, customer in-trip tracking and ETA |
| LFC | Partner onboarding, partner behaviour, waiting time policy, distance breach policy, commission policy, account suspension, timer delay, incentives, cancellations |
| HSC | Packers & movers, large vehicles (10ft and above) |
| TNS | Support, partner communications, notifications, payments, fraud |

### Problem type taxonomy
| Value | Meaning |
|-------|---------|
| `touchpoint_app` | A screen or interaction on the app (UI/UX) |
| `touchpoint_ops` | An interaction with a Porter representative |
| `service` | Underlying logic or algorithm (pricing engine, allocation logic, etc.) |
| `policy` | Base rules governing the services (thresholds, commission rates, etc.) |
| `service/policy` | Calculation seems wrong but unclear if logic or rule |
| `touchpoint_app/service` | Display looks wrong but unclear if screen or underlying data |
| `touchpoint_ops/policy` | Human decision seems wrong but unclear if rep or policy |

## Files

```
porter-clustering-classification-tickets/   (this folder)
├── run_workbench.ipynb         # Run steps 1-3 on Vertex AI Workbench
├── disaggregate.py             # Step 1: narrative → 7-field structured output (Gemini via Vertex AI)
├── embed.py                    # Step 2: Embedding (local, structured multi-field text)
├── cluster.py                  # Step 3: UMAP + HDBSCAN (local, supports --prob-threshold)
├── pipeline_utils.py           # Shared utilities
├── system_prompt.txt           # LLM instructions (Porter-specific, 7-field output, rubric + stage-level business context)
├── examples.txt                # Few-shot examples (18 Porter-specific examples, 7-field schema)
├── manual_run_guide.md         # Web UI guide for running analysis on ≤200 transcripts (Claude or Gemini)
├── gemini_guide.md             # Step-by-step Gemini walkthrough with disaggregation + clustering prompts
├── clustering_prompt.txt       # Standalone clustering prompt — open, replace PROBLEMS section, paste into Gemini
├── PROJECT_CONTEXT_SUPPORT_CLUSTERING.md
└── data/
    ├── *.csv                           # Input: output from batch_summarize.py
    ├── blueprint_porter - partner.csv  # Stage-level business context for partner app journeys
    ├── disaggregated.parquet           # After step 1
    ├── embedded.parquet                # After step 2
    └── clustered.parquet               # Final output
```

## Usage

### Prerequisites

1. **GCP Project** with Vertex AI enabled
2. **Authentication**: `gcloud auth application-default login`
3. **Python packages**:
   ```bash
   pip install google-cloud-aiplatform pandas sentence-transformers umap-learn hdbscan pyarrow
   ```

### Steps 1-3: Run individually

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id

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
python3 cluster.py data/embedded.parquet data/clustered.parquet \
    --min-cluster-size 15 \
    --prob-threshold 0.0
```

### Run on Vertex AI Workbench

Use `run_workbench.ipynb`. Edit the config cell:

```python
PROJECT_ID = "your-actual-project-id"
BUCKET = "your-bucket-name"
GCS_INPUT = "gs://your-bucket-name/inputs/tickets.csv"
LIMIT = 5000
WORKERS = 10
MIN_CLUSTER_SIZE = 15
PROB_THRESHOLD = 0.0    # 0.0 = keep all; 0.5 = drop low-confidence cluster assignments
REGION = "us-central1"
```

#### Authentication

No manual auth setup needed — Workbench VMs automatically use the instance's **default service account**, which already has access to Vertex AI and GCS buckets in the same project.

**Required IAM roles** for the Workbench service account:
- `Vertex AI User` (for Gemini API calls)
- `Storage Object Admin` (for GCS read/write)
- `Service Usage Consumer` (for API access)

#### Checkpoint/resume

If disaggregation fails partway through (API error, timeout, etc.), it saves a checkpoint after every batch of 50 rows. Run with `--resume` flag to pick up where it left off.

#### 429 Rate Limiting

Gemini 2.0 Flash has default quotas per project. If you hit 429 errors:
1. Reduce `--workers` (try 5, then 1)
2. Long-term fix: add exponential backoff retry logic to `disaggregate.py`
3. Quota increase: GCP Console → IAM & Admin → Quotas → search for "Generate content requests"

#### Stop Workbench when done

Go to **GCP Console → Vertex AI → Workbench** and click **STOP**. A running Workbench charges ~$8/day even when idle.

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

### Why UMAP before HDBSCAN?
- Reduces 384 dimensions to 15
- Makes clustering faster and more accurate
- Preserves semantic relationships

### Why 9 fields instead of 1?
- `summarised_problem` is the core clustering text
- `fidelity` lets you filter to only `pain_and_touchpoint` rows for deep investigation
- `journey`, `stage`, `team`, `problem_type` enable PM/leadership to quickly route and prioritise clusters without reading every problem statement; `stage` is more granular than `journey` and separates problems that share a journey but occur at different steps
- `mechanism` is a noun phrase naming the specific component that broke ("waiting time compensation", "allocation delay", "trip start button") — stripped of the failure verb; two problems with the same mechanism but different failure modes share this value and can be split at the clustering stage rather than merged
- `impact` lets you triage by severity — `blocked` and `financial_loss` rows surface first
- `cause_fidelity` signals whether the root cause is known (actionable immediately) or needs investigation
- All 9 fields are embedded together as a structured string — metadata provides hard structural separation between problems in the same touchpoint area
- Multi-problem rows are exploded into one row per problem, so the output is fully flat and filterable

## Scaling Notes

### 20K (current target)
- Works on small VM (e2-standard-4)
- ~1-2 hours total
- ~$11 cost

### 500K (future option)
- Same code, just takes longer
- ~8-12 hours total
- ~$270 cost

## Changes

### Claude Haiku → Gemini 2.0 Flash in disaggregate.py (Feb 2025)

**Why:** Claude Haiku on Vertex AI required third-party model quota approval. New GCP projects start with zero quota for third-party models and even a single API call was rejected with 429 errors.

**What changed:**
- `disaggregate.py`: Replaced `anthropic.AnthropicVertex` with `vertexai.GenerativeModel`
- Model: `claude-3-haiku@20240307` → `gemini-2.0-flash-001`
- Region: `us-east5` → `us-central1`
- JSON output: `response_mime_type="application/json"` instead of free-form text parsing
- Removed prompt caching logic (Gemini is cheap without it)
- Dependency: `anthropic[vertex]` → `google-cloud-aiplatform`

---

### IT context → Porter context (Feb 2026)

**Why:** Original pipeline built on IT support ticket sample data. Forked into `porter-clustering-tickets` for real Porter Play Store data.

**What changed:**
- Added pre-step: `batch_summarize.py` (Claude Haiku via Anthropic Batch API) for multilingual review → English narrative
- `system_prompt.txt`: Rewritten for Porter context (input = narrative, output = 1-field `summarised_user_problem`)
- `examples.txt`: 13 Porter-specific examples replacing IT ticket examples
- `disaggregate.py`: Output schema from 7 fields → 1 field

---

### Reorganisation: pre-step moved into porter-playstore-reviews/ (Feb 2026)

**Why:** Scraping and summarisation scripts previously lived in `~/`. Moved into dedicated subfolder for self-containment.

**What changed:**
- Created `porter-playstore-reviews/` inside `porter-clustering-tickets/`
- Moved `porter_reviews.py`, `batch_summarize.py`, prompt files, and data CSVs into it

---

### Fork: porter-clustering-classification-tickets (Feb 2026)

**Why:** Add structured classification metadata to disaggregation output. Enables PM/leadership routing without reading every cluster. Enables hybrid classification + clustering architecture for known vs unknown problem buckets.

**What changed:**

#### system_prompt.txt (full rewrite)
- Added formal problem definition: valid problem requires pain-point + specific touchpoint
- Added Porter partner journey context (Onboarding → Order Acceptance → Order Execution → Post-Order → Account & Compliance → Incentives → Customer Support → Notifications)
- Added Porter customer journey context (Authentication → Service Selection → Booking Details → Vehicle Selection → Order Review & Placement → Partner Allocation → Pre-Pickup → In-Trip → Trip Completion)
- Added cross-cutting journeys (Payments, Ratings, Cancellations)
- Added journey taxonomy (predefined list to assign `journey` from)
- Added team ownership table (CGE, Marketplace, LFC, HSC, TNS)
- Added problem type taxonomy (touchpoint_app, touchpoint_ops, service, policy, and hybrid values)
- Changed output from 1 field to 5 fields: `summarised_problem`, `fidelity`, `journey`, `team`, `problem_type`
- Added fidelity definition: `pain_only` vs `pain_and_touchpoint`
- Added multi-problem splitting rules: split 2-3 distinct extractable problems into array; flag more than 3 as "multiple problems - to be broken down"
- Added key guardrails:
  - Only assign `pain_and_touchpoint` when touchpoint explicitly stated or unambiguously implied — never infer
  - Do not attribute causality unless explicitly stated in the input
  - If review only describes resolution attempt (not underlying problem) → `pain_only`
  - If underlying problem clear AND support failure is part of pain → include specific support failure in statement
  - Use canonical abstract language — same problem should produce same output regardless of reviewer wording
  - When listing journeys, do not default to journey/domain name when touchpoint is not listed — name it specifically
- Added Distinguishing Output Types section clarifying when each fallback applies
- Added "What Makes a Good Touchpoint" section with too-broad vs right-level examples

#### examples.txt (full rewrite)
- 12 Porter-specific examples in 5-field format
- Covers: pain_only, pain_and_touchpoint, insufficient information, no actionable information, multiple problems - to be broken down
- Example 10 is a split (array of 2 problems, two different teams: Marketplace + TNS)
- Key examples:
  1. Order frequency dropped → `pain_only`, `journey: Order Acceptance`, `team: unknown` (cause not identifiable)
  2. Scooty/bike same fare → `pain_and_touchpoint`, `journey: Order Acceptance`, `team: Marketplace`, `problem_type: policy`
  3. Partner issues not resolved → `pain_only`, `journey: Customer Support`, `team: TNS`, `problem_type: touchpoint_ops`
  4. App speed slow → `pain_only`, `team: unknown`, `problem_type: touchpoint_app`
  5. Multiple technical issues → `multiple problems - to be broken down`
  6. Waiting time charges not collectible → `pain_and_touchpoint`, `journey: Order Execution`, `team: LFC`, `problem_type: service/policy`
  7. Neither good nor bad → `insufficient information`
  8. Protest warning → `no actionable information`
  9. Account suspended without investigation → `pain_and_touchpoint`, `journey: Account & Compliance`, `team: LFC`, `problem_type: touchpoint_ops/policy`
  10. Fare discrepancy + support not mediating → split array: `[Payments/Marketplace/service/policy, Customer Support/TNS/touchpoint_ops]`
  11. Useless app rant → `no actionable information`
  12. Last-minute cancellations → `pain_and_touchpoint`, `journey: Cancellations`, `team: LFC`, `problem_type: service/policy`

#### disaggregate.py
- `disaggregate_one` return structure changed to wrapper dict with `problems` list (always a list, even for single problems)
- Error fallback dicts updated to all 5 fields
- Merge loop updated to explode multi-problem rows — one output row per problem per original row
- New fields added to merged row: `fidelity`, `journey`, `team`, `problem_type`

#### embed.py
- Column reference changed: `summarised_user_problem` → `summarised_problem`
- Skip filter updated to include new fallback values:
  ```python
  SKIP_VALUES = {
      'NA', 'PARSE_ERROR', 'API_ERROR', 'ERROR',
      'insufficient information', 'no actionable information',
      'multiple problems - to be broken down'
  }
  ```

#### cluster.py
- Added `--prob-threshold` argument (default 0.0, float between 0 and 1)
- After HDBSCAN fit, any point with `clusterer.probabilities_ < prob_threshold` is reassigned to noise (-1)
- Useful lever to improve cluster purity without changing min_cluster_size

#### run_workbench.ipynb
- Added `PROB_THRESHOLD = 0.0` to config cell
- Updated cluster command to pass `--prob-threshold {PROB_THRESHOLD}`
- Added `Prob threshold` to config print output
- Fixed IndentationError: cluster cell command collapsed to single line (backslash continuation does not work in Jupyter `!` magic commands)

---

### Schema expansion: impact + cause_fidelity fields; metadata embedding (Feb 2026)

**Why:** Two new fields added to improve clustering accuracy and triage utility. Metadata embedding ensures structural differences between problems surface in the embedding space, not just semantic similarity of the problem text.

**What changed:**

#### disaggregate.py
- Added `impact` and `cause_fidelity` to all three error fallback dicts (PARSE_ERROR, API_ERROR, thread error)
- Thread error fallback corrected: was using old `summarised_user_problem` key with flat structure; now uses correct `problems` list wrapper
- Merge loop updated to extract both new fields from each problem object:
  ```python
  merged['impact'] = problem.get('impact', 'unknown')
  merged['cause_fidelity'] = problem.get('cause_fidelity', 'N/A')
  ```

#### embed.py
- Replaced single-field embedding (`summarised_problem` only) with structured multi-field text
- New `build_embedding_text` function combines all 7 fields into a pipe-separated string, omitting fields with `unknown` or `N/A` values:
  ```python
  def build_embedding_text(row):
      parts = [row['summarised_problem']]
      for field in ['fidelity', 'impact', 'journey', 'team', 'problem_type', 'cause_fidelity']:
          val = str(row.get(field, 'unknown'))
          if val not in ('unknown', 'N/A', ''):
              parts.append(val)
      return ' | '.join(parts)
  ```
- Sample embedding text is logged on first run for inspection

#### system_prompt.txt
- Added `feature_request` as a third fidelity level — for specific, bounded new capabilities with no existing step failing
- Added `impact` field with four values: `financial_loss`, `blocked`, `inconvenienced`, `informative`
- Added `cause_fidelity` field: `cause_known` (trigger explicitly stated), `cause_unknown` (failure mode identifiable but trigger not stated), `N/A` (for pain_only and feature_request rows)
- Output Format updated to show all 7 fields across all output type examples
- "What Qualifies as a Problem" restructured into three explicit categories with guardrails on feature_request specificity
- Failure mode encoding rule updated: encode when clear, describe outcome neutrally when ambiguous — never guess
- Failure mode table retains 11 modes; a second comparison table added showing clear vs ambiguous examples
- Added condition instruction: include the circumstance that triggered the failure when it differentiates problems on the same touchpoint
- Added verb floor: every problem statement must contain a verb; noun phrases alone are invalid
- Added second self-check: "is there any detail in the review that would differentiate this from a similar problem on the same touchpoint?"
- Anti-over-abstraction guardrail and existing self-check preserved

#### examples.txt
- All 18 examples updated to include `impact` and `cause_fidelity` fields
- Examples 13–16 added as contrastive pairs showing failure mode distinction:
  - 13 vs 14: same touchpoint (document verification), different failure modes (delayed vs no feedback)
  - 15 vs 16: same touchpoint (waiting time), different failure modes (not triggered vs incorrectly triggered)
- Examples 17–18 added for feature_request fidelity:
  - 17: geographic expansion request → `feature_request`, `impact: inconvenienced`
  - 18: multiple simultaneous booking → `feature_request`, `journey: Order Acceptance`, `team: Marketplace`

---

### System prompt: business context rebuilt from blueprint; journey taxonomy updated (Feb 2026)

**Why:** The previous business context described journeys at a high level. The blueprint (`data/blueprint_porter - partner.csv`) provides stage-level descriptions with specific system behaviours, edge cases, and gray areas. Using the blueprint as the primary source gives the LLM the vocabulary and context needed to produce correct touchpoint names and set `cause_fidelity` accurately.

**What changed:**

#### system_prompt.txt — Partner Side business context (full replacement)
- Replaced high-level paragraph descriptions with stage-level structure mirroring the blueprint exactly
- Each journey section now lists its stages with one-line descriptions of what happens, what can fail, and where gray areas exist
- Gray area stages noted explicitly — these typically produce `cause_fidelity: cause_unknown` since the specific circumstances can't be confirmed from a review alone
- New touchpoint vocabulary now available to the LLM: `document upload`, `document approval`, `stage 2 documents`, `training videos`, `dummy order`, `registration fee`, `waitlisting fee`, `allocation delay`, `allocation logic`, `fare calculation for notification`, `missed-order penalty`, `Prime status`, `coordinate loading / mark arrived at pickup`, `edit order`, `distance breach`, `coordinate drop / mark arrived at drop`, `trip fare calculation`, `payment collection`, definite vs indefinite `suspension`

#### system_prompt.txt — Journey taxonomy updated
| Before | After | Reason |
|---|---|---|
| (not present) | `Order Allocation` | Blueprint separates allocation (notification, delay, logic) from acceptance (tap to accept) |
| `Post-Order` | `Post Trip` | Matches blueprint terminology |
| `Incentives` | `Performance` | Matches blueprint section name |
| `Notifications` | (removed) | Folded into Customer Support per blueprint |
| (not present) | `Profile` | Blueprint has a separate edit details stage |

#### system_prompt.txt — "What Makes a Good Touchpoint" table
- Updated all right-level examples to use actual blueprint stage names rather than generic descriptions

#### examples.txt
- Example 1: `journey` updated `Order Acceptance` → `Order Allocation` (order frequency drop is an allocation-level problem)
- Example 2: `journey` updated `Order Acceptance` → `Order Allocation` (fare calculation for notification lives in Order Allocation)

---

### New files: web UI guides for manual analysis on ≤200 transcripts (Feb 2026)

**Why:** For small batches or proof-of-concept runs, the full pipeline is unnecessary overhead. Claude Projects and Gemini Gems can run disaggregation and clustering directly at this scale.

#### manual_run_guide.md (new)
- Setup instructions for Claude Projects and Gemini Gems
- Disaggregation batch prompt template (process 20–30 transcripts per conversation)
- Google Sheets formula to format transcripts for copy-paste
- Full clustering prompt with: positive definition of a cluster, "never merge" rules, same-fix self-check, naming format with good/bad examples table
- Follow-up prompt templates for fixing vague names, broad clusters, wrong merges, and duplicate clusters
- Decision table: when to use web UI vs pipeline

#### gemini_guide.md (new)
- Step-by-step Gemini-specific walkthrough (Gem setup, batch disaggregation, data preparation, clustering, review and fix-up)
- Full clustering prompt inline — no need to cross-reference other files
- Example file column spec for uploading real Porter examples as supplementary Gem context: `body`, `summarised_problem`, `fidelity`, `journey`, `team`, `problem_type`, `impact`, `cause_fidelity`, `notes` (optional, for non-obvious cases only)
