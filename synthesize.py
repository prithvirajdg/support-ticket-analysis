"""
Synthesize rich sub-problem statements per cluster from review bodies.
Runs after cluster.py. Processes all clusters including noise (cluster == -1).

For each cluster, passes all review narratives (body column) to Gemini and asks it to:
  1. Identify distinct sub-scenarios within the cluster
  2. Write a rich 2-3 sentence problem statement per sub-scenario
  3. Assign each review to exactly one sub-scenario

Noise cluster (-1): treated as an exploratory pass — finds patterns if any exist;
if reviews are genuinely disparate, each becomes its own sub-scenario.

Output: one row per sub-scenario with count, proportion, review_ids, and quality_flag.

Usage:
    python3 synthesize.py data/clustered.parquet data/cluster_synthesis.csv \\
        --project your-project-id \\
        --region us-central1 \\
        --workers 5

    # Skip very small non-noise clusters
    python3 synthesize.py data/clustered.parquet data/cluster_synthesis.csv \\
        --project your-project-id --min-cluster-size 5

    # Resume from checkpoint if job failed
    python3 synthesize.py data/clustered.parquet data/cluster_synthesis.csv \\
        --project your-project-id --resume
"""

import os
import json
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "gemini-2.0-flash-001"
DEFAULT_REGION = "us-central1"
CHECKPOINT_SUFFIX = ".synthesis_checkpoint.json"

# Phrases that indicate a vague rich_problem (checked lowercase, startswith)
VAGUE_OPENINGS = (
    'various', 'multiple', 'issues with', 'problems with', 'there are',
    'partners face', 'customers face', 'users face', 'the app has',
    'there is a', 'some partners', 'some customers',
)
MIN_RICH_PROBLEM_LENGTH = 80  # characters — 2-3 sentences should comfortably exceed this

# =============================================================================
# SYNTHESIS PROMPTS
# =============================================================================

# Shared guardrails appended to both prompts
_GUARDRAILS = """\
Guardrails for each rich_problem:
- Must name a specific touchpoint, feature, or policy — not a journey name (e.g. "Order \
Execution") or a team name (e.g. "Marketplace"). Name the exact screen, button, timer, \
calculation, or rule that failed.
- Must contain a failure verb (e.g. not triggered, delayed, blocked, incorrectly applied, \
not credited, not shown, stalling).
- Must state the direct consequence to the user (money lost, blocked from working, unable \
to proceed, no recourse available, etc.).
- Must not be a vague statement like "issues with X", "problems with Y", "X doesn't work \
properly", or "partners face difficulties with Z".
- Must add significant new information beyond the cluster theme. Do not restate or slightly \
expand the theme. The rich_problem must include at least one of: (a) the specific trigger \
condition or circumstance under which the failure occurs, (b) why the system fails at that \
touchpoint, or (c) a consequence detailed enough to distinguish this sub-scenario from the \
theme alone. The exact values of thresholds or parameters do not need to be precise — \
describe the condition, not the number.

  BAD (restatement): "Partners report that waiting time compensation is not triggered. \
This happens during trips and partners lose money."
  GOOD: "When a partner marks arrived at pickup but their GPS position does not match the \
customer's pickup pin closely enough at the moment of tapping, the system does not register \
the mark-arrived action as valid and the waiting time clock never starts. Partners wait \
without any compensation and have no in-app mechanism to flag or dispute the missed clock \
start."

- Self-check before returning: read each rich_problem and verify it (1) names a specific \
touchpoint, (2) contains a failure verb, (3) states a direct user consequence, and \
(4) adds information not already present in the cluster theme. Rewrite any that fail \
before returning the JSON.
"""

SYNTHESIS_PROMPT = """\
You are analyzing a cluster of user reviews that share the same underlying problem theme.

Cluster theme: {cluster_problem}

Below are {n} user review narratives from this cluster, numbered [1] to [{n}].

{numbered_bodies}

---

Task:
1. Identify the distinct sub-scenarios within this cluster. A sub-scenario has a different \
root cause, trigger condition, or failure path — even if the surface symptom is the same. \
For example, "customer refuses to pay" could split into: \
(a) customer disputes a fare increase they were not informed about because the distance \
breach was not surfaced to them during the trip, \
(b) customer is unreachable or stalling after trip completion and support does not \
intervene to compel payment.

2. For each sub-scenario, write a rich problem statement of 2-3 sentences covering: \
what specifically fails or goes wrong, under what condition or trigger, and what the user \
impact is.

3. Assign each review number [1]–[{n}] to exactly one sub-scenario. Every review must be \
assigned. No review may appear in more than one sub-scenario.

4. Do not create more than 5 sub-scenarios. If reviews share the same root cause and would \
need the same fix, group them together even if the surface wording differs.

{guardrails}
Return a JSON array. Each element must have exactly these two keys:
  "rich_problem": the 2-3 sentence problem statement
  "review_ids": array of integer review numbers assigned to this sub-scenario

Example format:
[
  {{"rich_problem": "When a partner marks arrived at pickup before the customer confirms their location, the waiting time clock starts but the compensation is not credited at trip end. This occurs because the system treats the mark-arrived action as unilateral regardless of whether the customer acknowledged the partner's arrival. Partners lose waiting time earnings with no visible record of the dispute.", "review_ids": [1, 3, 7]}},
  {{"rich_problem": "...", "review_ids": [2, 4, 5, 6]}}
]
"""

SYNTHESIS_PROMPT_NOISE = """\
You are analyzing a set of user reviews that did not form a stable cluster. They may share \
some recurring patterns, or they may be genuinely disparate problems with nothing in common.

Below are {n} user review narratives, numbered [1] to [{n}].

{numbered_bodies}

---

Task:
1. Look for any recurring patterns, shared root causes, or similar failure conditions among \
these reviews. If patterns exist, group them into sub-scenarios.

2. If some reviews appear genuinely unrelated — different touchpoints, different failures, \
no common thread — each distinct problem can be its own sub-scenario.

3. For each sub-scenario, write a rich problem statement of 2-3 sentences covering: \
what specifically fails or goes wrong, under what condition or trigger, and what the user \
impact is.

4. Assign each review number [1]–[{n}] to exactly one sub-scenario. Every review must be \
assigned. No review may appear in more than one sub-scenario.

5. Do not create more than 10 sub-scenarios. Group reviews that share the same root cause \
and fix together even if wording differs.

{guardrails}
Return a JSON array. Each element must have exactly these two keys:
  "rich_problem": the 2-3 sentence problem statement
  "review_ids": array of integer review numbers assigned to this sub-scenario
"""

# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

def get_checkpoint_path(output_path: str) -> str:
    return output_path.replace('.csv', CHECKPOINT_SUFFIX).replace('.parquet', CHECKPOINT_SUFFIX)


def save_checkpoint(checkpoint_path: str, done_cluster_ids: list, rows: list):
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'done_cluster_ids': done_cluster_ids,
            'rows': rows,
            'timestamp': time.time()
        }, f)


def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def delete_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_dominant(series):
    """Return the mode of a Series, or 'unknown' if empty."""
    counts = series.value_counts()
    if len(counts) == 0:
        return 'unknown'
    return counts.index[0]


_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
    'through', 'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'yet',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'also', 'as', 'if', 'then', 'than', 'because', 'while', 'although',
    'after', 'before', 'during', 'without', 'between', 'against',
}

# Minimum number of new content words rich_problem must introduce beyond the cluster theme
MIN_NEW_CONTENT_WORDS = 10


def validate_rich_problem(text: str, cluster_problem: str) -> str:
    """
    Check a rich_problem string against basic quality criteria.
    Returns a flag string describing the issue, or '' if it passes.

    Checks (in order):
      too_short              — under MIN_RICH_PROBLEM_LENGTH characters
      vague_opening          — starts with known vague phrases
      insufficient_new_info  — adds fewer than MIN_NEW_CONTENT_WORDS content words
                               beyond what is already in the cluster theme
    """
    if not text or text.startswith('SYNTHESIS_ERROR'):
        return ''  # error rows are already marked separately

    if len(text) < MIN_RICH_PROBLEM_LENGTH:
        return 'too_short'

    lowered = text.lower()
    if any(lowered.startswith(v) for v in VAGUE_OPENINGS):
        return 'vague_opening'

    # Check that rich_problem adds substantial new content beyond the cluster theme.
    # We compare content words (non-stopwords) — if fewer than MIN_NEW_CONTENT_WORDS
    # are unique to rich_problem, it is likely a restatement or minor expansion.
    if cluster_problem and cluster_problem != 'unclustered (noise)':
        theme_words = set(cluster_problem.lower().split()) - _STOP_WORDS
        rich_words = set(lowered.split()) - _STOP_WORDS
        new_words = rich_words - theme_words
        if len(new_words) < MIN_NEW_CONTENT_WORDS:
            return 'insufficient_new_info'

    return ''


def synthesize_cluster(cluster_id, cluster_df, model, max_reviews, logger, is_noise=False):
    """
    Synthesize rich sub-problem statements for a single cluster.
    Returns a list of row dicts — one per identified sub-scenario.

    is_noise=True: uses the noise-specific prompt (no shared theme assumed).
    Proportion for noise is relative to sample_size, not cluster_size.
    """
    from vertexai.generative_models import GenerationConfig

    cluster_size = len(cluster_df)

    # Sample if cluster exceeds max_reviews
    if cluster_size > max_reviews:
        sample_df = cluster_df.sample(max_reviews, random_state=42)
    else:
        sample_df = cluster_df.copy()

    sample_size = len(sample_df)

    # Cluster problem label
    if is_noise:
        cluster_problem = 'unclustered (noise)'
    else:
        cluster_problem = get_dominant(cluster_df['summarised_problem'])

    # Dominant metadata from full cluster
    meta = {
        'team':         get_dominant(cluster_df['team'])         if 'team'         in cluster_df.columns else 'unknown',
        'journey':      get_dominant(cluster_df['journey'])      if 'journey'      in cluster_df.columns else 'unknown',
        'mechanism':    get_dominant(cluster_df['mechanism'])    if 'mechanism'    in cluster_df.columns else 'unknown',
        'failure_mode': get_dominant(cluster_df['failure_mode']) if 'failure_mode' in cluster_df.columns else 'unknown',
        'impact':       get_dominant(cluster_df['impact'])       if 'impact'       in cluster_df.columns else 'unknown',
    }

    # Build identifier list — use reviewId column if present, else DataFrame index
    id_col = 'reviewId' if 'reviewId' in sample_df.columns else None
    sample_ids = sample_df[id_col].tolist() if id_col else sample_df.index.tolist()

    # Build numbered bodies for prompt
    bodies = sample_df['body'].fillna('').tolist()
    numbered_bodies = '\n\n'.join([f"[{i + 1}] {body}" for i, body in enumerate(bodies)])

    # Select prompt template
    if is_noise:
        prompt = SYNTHESIS_PROMPT_NOISE.format(
            n=len(bodies),
            numbered_bodies=numbered_bodies,
            guardrails=_GUARDRAILS,
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(
            cluster_problem=cluster_problem,
            n=len(bodies),
            numbered_bodies=numbered_bodies,
            guardrails=_GUARDRAILS,
        )

    def error_row(error_msg):
        return {
            'cluster_id':      cluster_id,
            'cluster_size':    cluster_size,
            'sample_size':     sample_size,
            'cluster_problem': cluster_problem,
            'sub_problem_id':  1,
            'rich_problem':    f'SYNTHESIS_ERROR: {error_msg[:120]}',
            'count':           sample_size,
            'proportion':      1.0,
            'review_ids':      '',
            'is_noise':        is_noise,
            'quality_flag':    '',
            **meta,
        }

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        sub_problems = json.loads(raw)
        if not isinstance(sub_problems, list):
            sub_problems = [sub_problems]

    except json.JSONDecodeError as e:
        logger.error(f"  Cluster {cluster_id}: JSON parse error — {e}")
        return [error_row(f"JSON parse error: {e}")]
    except Exception as e:
        logger.error(f"  Cluster {cluster_id}: API error — {e}")
        return [error_row(f"API error: {e}")]

    # Map Gemini's 1-indexed review_ids back to actual review identifiers
    rows = []
    for i, sp in enumerate(sub_problems):
        raw_ids = sp.get('review_ids', [])
        actual_ids = []
        for pos_1indexed in raw_ids:
            try:
                pos = int(pos_1indexed) - 1
                if 0 <= pos < len(sample_ids):
                    actual_ids.append(str(sample_ids[pos]))
            except (ValueError, TypeError):
                pass

        count = len(actual_ids)
        # For noise: proportion is relative to sample (can't claim coverage of full noise set)
        # For regular clusters: proportion is relative to full cluster size
        denom = sample_size if is_noise else cluster_size
        proportion = round(count / denom, 3) if denom > 0 else 0.0

        rich_problem = sp.get('rich_problem', '')
        quality_flag = validate_rich_problem(rich_problem, cluster_problem)

        rows.append({
            'cluster_id':      cluster_id,
            'cluster_size':    cluster_size,
            'sample_size':     sample_size,
            'cluster_problem': cluster_problem,
            'sub_problem_id':  i + 1,
            'rich_problem':    rich_problem,
            'count':           count,
            'proportion':      proportion,
            'review_ids':      ', '.join(actual_ids),
            'is_noise':        is_noise,
            'quality_flag':    quality_flag,
            **meta,
        })

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Synthesize rich sub-problem statements per cluster from review bodies'
    )
    parser.add_argument('input',  help='Clustered Parquet/CSV file (output of cluster.py)')
    parser.add_argument('output', help='Output CSV — one row per sub-problem')
    parser.add_argument('--project', default=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                        help='GCP project ID (default: GOOGLE_CLOUD_PROJECT env var)')
    parser.add_argument('--region', default=DEFAULT_REGION,
                        help=f'Vertex AI region (default: {DEFAULT_REGION})')
    parser.add_argument('--workers', type=int, default=5,
                        help='Concurrent Gemini calls (default: 5)')
    parser.add_argument('--max-reviews', type=int, default=100,
                        help='Max review bodies to pass per cluster; larger clusters are sampled (default: 100)')
    parser.add_argument('--min-cluster-size', type=int, default=0,
                        help='Skip non-noise clusters smaller than this (default: 0 = process all). Noise is always processed.')
    parser.add_argument('--skip-noise', action='store_true',
                        help='Skip the noise cluster (cluster == -1) entirely')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    args = parser.parse_args()

    if not args.project:
        print("ERROR: No GCP project specified.")
        print("Either set GOOGLE_CLOUD_PROJECT environment variable or pass --project")
        sys.exit(1)

    import pandas as pd
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from pipeline_utils import setup_logging, validate_columns, StepTracker, ValidationError

    logger = setup_logging("synthesize")
    tracker = StepTracker("SYNTHESIS (Gemini per cluster)", logger)

    checkpoint_path = get_checkpoint_path(args.output)

    try:
        tracker.start(f"Synthesizing rich sub-problems from cluster review bodies ({MODEL})")
        logger.info(f"Project: {args.project}")
        logger.info(f"Region:  {args.region}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Max reviews per cluster: {args.max_reviews}")

        # Load input
        if not os.path.exists(args.input):
            raise ValidationError(f"Input file not found: {args.input}")

        logger.info(f"Loading {args.input}...")
        df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")

        validate_columns(df, ['cluster', 'summarised_problem', 'body'], "synthesize", logger)
        tracker.checkpoint("Data loaded and validated", len(df))

        noise_count = (df['cluster'] == -1).sum()
        named_count = (df['cluster'] != -1).sum()
        logger.info(f"Named cluster rows: {named_count} | Noise rows: {noise_count}")

        # Apply min-cluster-size filter to non-noise clusters only
        all_df = df.copy()
        if args.min_cluster_size > 0:
            non_noise = all_df[all_df['cluster'] != -1]
            sizes = non_noise.groupby('cluster').size()
            valid = sizes[sizes >= args.min_cluster_size].index
            before = non_noise['cluster'].nunique()
            kept = non_noise[non_noise['cluster'].isin(valid)]
            dropped = before - kept['cluster'].nunique()
            noise_df = all_df[all_df['cluster'] == -1]
            all_df = pd.concat([kept, noise_df], ignore_index=True)
            logger.info(f"Named clusters after min-size filter ({args.min_cluster_size}): "
                        f"{kept['cluster'].nunique()} (dropped {dropped})")

        # Build list of cluster IDs to process
        cluster_ids = sorted(all_df[all_df['cluster'] != -1]['cluster'].unique())
        if not args.skip_noise and noise_count > 0:
            cluster_ids = [-1] + list(cluster_ids)  # process noise first
        elif args.skip_noise:
            logger.info("Skipping noise cluster (--skip-noise)")

        logger.info(f"Total clusters to process: {len(cluster_ids)} "
                    f"(including noise: {not args.skip_noise and noise_count > 0})")

        # Init Vertex AI
        logger.info(f"Initializing Vertex AI (project={args.project}, region={args.region})...")
        vertexai.init(project=args.project, location=args.region)
        model = GenerativeModel(MODEL)
        logger.info("Vertex AI initialized")
        tracker.checkpoint("Vertex AI initialized")

        # Checkpoint resume
        done_cluster_ids = set()
        all_rows = []

        if args.resume:
            cp = load_checkpoint(checkpoint_path)
            if cp:
                done_cluster_ids = set(cp['done_cluster_ids'])
                all_rows = cp['rows']
                logger.info(f"Resuming: {len(done_cluster_ids)} clusters already done")
            else:
                logger.info("No checkpoint found, starting from beginning")

        remaining = [cid for cid in cluster_ids if cid not in done_cluster_ids]
        total = len(cluster_ids)
        completed = len(done_cluster_ids)

        logger.info(f"Total clusters: {total} | Remaining: {len(remaining)}")

        start_time = time.time()

        def process_cluster(cluster_id):
            cluster_df = all_df[all_df['cluster'] == cluster_id].copy()
            is_noise = (cluster_id == -1)
            return cluster_id, synthesize_cluster(
                cluster_id, cluster_df, model, args.max_reviews, logger, is_noise=is_noise
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_cluster, cid): cid for cid in remaining}

            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    _, rows = future.result()
                    all_rows.extend(rows)
                    done_cluster_ids.add(cluster_id)
                    completed += 1

                    cluster_size = all_df[all_df['cluster'] == cluster_id].shape[0]
                    flagged = sum(1 for r in rows if r.get('quality_flag'))
                    flag_note = f" | {flagged} flagged" if flagged else ""
                    noise_note = " [NOISE]" if cluster_id == -1 else ""
                    logger.info(f"  [{completed}/{total}] Cluster {cluster_id}{noise_note}: "
                                f"{len(rows)} sub-problem(s) from {cluster_size} reviews{flag_note}")

                    # Save checkpoint every 10 clusters
                    if completed % 10 == 0:
                        save_checkpoint(checkpoint_path, list(done_cluster_ids), all_rows)
                        logger.info(f"  Checkpoint saved ({completed} clusters done)")

                except Exception as e:
                    logger.error(f"  Cluster {cluster_id} failed: {e}")

        # Final save
        result_df = pd.DataFrame(all_rows)
        if not result_df.empty and 'cluster_id' in result_df.columns:
            result_df = result_df.sort_values(['cluster_id', 'sub_problem_id']).reset_index(drop=True)

        result_df.to_csv(args.output, index=False)
        delete_checkpoint(checkpoint_path)
        logger.info("Checkpoint cleared (job complete)")

        tracker.complete(args.output, len(result_df))

        # Summary
        elapsed = time.time() - start_time
        flagged_total = result_df['quality_flag'].astype(bool).sum() if 'quality_flag' in result_df.columns else 0

        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Clusters processed: {total}")
        logger.info(f"Total sub-problems: {len(result_df)}")
        if total > 0:
            logger.info(f"Avg sub-problems per cluster: {len(result_df) / total:.1f}")
        if flagged_total > 0:
            logger.warning(f"Quality flags: {flagged_total} rows flagged — review quality_flag column")
        logger.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
        logger.info(f"Output: {args.output}")
        logger.info(f"{'='*60}")
        logger.info(f"\nNext step: open {args.output}, filter quality_flag != '' first, "
                    f"then review and edit rich_problem per cluster.")

    except Exception as e:
        tracker.fail(e)
        logger.error(f"Job failed! Checkpoint saved at: {checkpoint_path}")
        logger.error("To resume, run with --resume flag")
        raise


if __name__ == '__main__':
    main()
