"""
Disaggregate support transcripts using Gemini via Vertex AI.
Runs on Compute Engine VM or Workbench with concurrent API calls.

Features:
- JSON mode (response_mime_type) for reliable structured output
- Checkpoint/resume capability (saves progress every batch)
- Concurrent API calls for speed

Usage:
    # Single file
    python3 disaggregate.py --input data/tickets.csv --output data/disaggregated.parquet

    # Resume from checkpoint if job failed
    python3 disaggregate.py --input data/tickets.csv --output data/disaggregated.parquet --resume

    # Limit for testing
    python3 disaggregate.py --input data/tickets.csv --output data/disaggregated.parquet --limit 100

Input: CSV/Parquet with 'body' column
Output: Parquet with original columns + disaggregated fields

Prerequisites:
    - Google Cloud SDK installed and authenticated (gcloud auth application-default login)
    - GOOGLE_CLOUD_PROJECT environment variable set, or pass --project
    - pip install google-cloud-aiplatform
"""

import os
import json
import time
import argparse
import sys
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gemini model on Vertex AI
MODEL = "gemini-2.0-flash-001"         # Cheap, fast, good for structured extraction
# MODEL = "gemini-2.0-flash-lite-001"  # Even cheaper, use if Flash quality is sufficient

# Default GCP region for Vertex AI Gemini
# Gemini is available in many regions; us-central1 is the most common
DEFAULT_REGION = "us-central1"


# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

def get_checkpoint_path(output_path: str) -> str:
    """Get checkpoint file path based on output path."""
    return output_path.replace('.parquet', '.checkpoint.json').replace('.csv', '.checkpoint.json')


def save_checkpoint(checkpoint_path: str, processed_rows: int, results: list, stats: dict):
    """Save progress to checkpoint file."""
    checkpoint = {
        'processed_rows': processed_rows,
        'results': results,
        'stats': stats,
        'timestamp': time.time()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def delete_checkpoint(checkpoint_path: str):
    """Delete checkpoint file after successful completion."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_vertex_client(project_id: str, region: str, system_prompt: str, examples: str):
    """
    Create a Gemini GenerativeModel configured for Vertex AI.

    Authentication: Uses Application Default Credentials (ADC).
    Run 'gcloud auth application-default login' locally, or
    use the default service account on Compute Engine/Workbench.
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=project_id, location=region)

    system_instruction = f"{system_prompt}\n\n{examples}"

    model = GenerativeModel(
        MODEL,
        system_instruction=system_instruction,
    )
    return model


def disaggregate_one(
    model,
    transcript: str,
    system_prompt: str,
    examples: str,
    model_name: str,
    row_idx: int
) -> dict:
    """
    Process a single transcript with Gemini via Vertex AI.

    Uses response_mime_type="application/json" to force valid JSON output,
    reducing parse errors compared to free-form text responses.
    """
    from vertexai.generative_models import GenerationConfig

    try:
        response = model.generate_content(
            f"SUMMARY:\n{transcript}\n\nOUTPUT:",
            generation_config=GenerationConfig(
                max_output_tokens=1024,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        result_text = response.text.strip()

        # Get token usage stats
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

        try:
            parsed = json.loads(result_text)
            # Normalise to always be a list of problems
            problems = parsed if isinstance(parsed, list) else [parsed]
            return {
                'problems': problems,
                '_input_tokens': input_tokens,
                '_output_tokens': output_tokens,
                '_row_idx': row_idx,
                '_error': None
            }
        except json.JSONDecodeError as e:
            return {
                'problems': [{"summarised_problem": "PARSE_ERROR", "fidelity": "unknown", "journey": "unknown", "stage": "unknown", "mechanism": "unknown", "failure_mode": "unknown", "team": "unknown", "problem_type": "unknown", "impact": "unknown", "cause_fidelity": "unknown"}],
                "_raw_response": result_text,
                "_input_tokens": input_tokens,
                "_output_tokens": output_tokens,
                "_row_idx": row_idx,
                "_error": f"JSON parse error: {e}"
            }
    except Exception as e:
        return {
            'problems': [{"summarised_problem": "API_ERROR", "fidelity": "unknown", "journey": "unknown", "stage": "unknown", "mechanism": "unknown", "failure_mode": "unknown", "team": "unknown", "problem_type": "unknown", "impact": "unknown", "cause_fidelity": "unknown"}],
            "_row_idx": row_idx,
            "_error": f"API error: {str(e)}"
        }


def process_batch_concurrent(
    client,
    transcripts: list,
    system_prompt: str,
    examples: str,
    model: str,
    start_idx: int,
    max_workers: int,
    logger
) -> list:
    """
    Process a batch of transcripts concurrently using ThreadPoolExecutor.

    This replaces Modal's .map() functionality with standard Python concurrency.
    """
    results = [None] * len(transcripts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                disaggregate_one,
                client,
                transcript,
                system_prompt,
                examples,
                model,
                start_idx + i
            ): i
            for i, transcript in enumerate(transcripts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Task {start_idx + idx} failed: {e}")
                results[idx] = {
                    'problems': [{"summarised_problem": "API_ERROR", "fidelity": "unknown", "journey": "unknown", "stage": "unknown", "mechanism": "unknown", "failure_mode": "unknown", "team": "unknown", "problem_type": "unknown", "impact": "unknown", "cause_fidelity": "unknown"}],
                    "_row_idx": start_idx + idx,
                    "_error": f"Thread error: {str(e)}"
                }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Disaggregate support transcripts using Gemini via Vertex AI'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV/Parquet file(s) - supports glob patterns')
    parser.add_argument('--output', required=True,
                        help='Output Parquet file')
    parser.add_argument('--project', default=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                        help='GCP Project ID (default: GOOGLE_CLOUD_PROJECT env var)')
    parser.add_argument('--region', default=DEFAULT_REGION,
                        help=f'Vertex AI region (default: {DEFAULT_REGION})')
    parser.add_argument('--workers', type=int, default=10,
                        help='Concurrent API calls (default: 10)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for progress tracking (default: 50)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max transcripts to process, 0 = all (default: 0)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    args = parser.parse_args()

    # Validate project
    if not args.project:
        print("ERROR: No GCP project specified.")
        print("Either set GOOGLE_CLOUD_PROJECT environment variable or pass --project")
        sys.exit(1)

    import pandas as pd
    from pipeline_utils import setup_logging, validate_columns, StepTracker, ValidationError

    logger = setup_logging("disaggregate")
    tracker = StepTracker("DISAGGREGATION", logger)

    # Checkpoint setup
    checkpoint_path = get_checkpoint_path(args.output)
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            logger.info(f"Found checkpoint: {checkpoint['processed_rows']} rows already processed")
        else:
            logger.info("No checkpoint found, starting from beginning")

    try:
        tracker.start(f"Processing transcripts with Gemini via Vertex AI ({MODEL})")
        logger.info(f"Project: {args.project}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Concurrent workers: {args.workers}")

        # Find input files
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if '*' in args.input:
            input_files = glob(args.input)
        else:
            input_files = [args.input]

        if not input_files:
            raise ValidationError(f"No input files found matching: {args.input}")

        logger.info(f"Found {len(input_files)} input file(s)")
        for f in input_files:
            logger.info(f"  - {f}")

        tracker.checkpoint("Input files validated", len(input_files))

        # Load and combine input files
        dfs = []
        for f in input_files:
            logger.info(f"Loading {f}...")
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            logger.info(f"  Loaded {len(df)} rows")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        logger.info(f"Total rows: {len(df)}")

        # Validate required column
        validate_columns(df, ['body'], "disaggregate", logger)
        tracker.checkpoint("Data loaded and validated", len(df))

        # Load prompts
        with open(os.path.join(script_dir, "system_prompt.txt"), "r") as f:
            system_prompt = f.read()
        with open(os.path.join(script_dir, "examples.txt"), "r") as f:
            examples = f.read()

        logger.info("Prompts loaded")

        # Create Vertex AI client
        logger.info("Initializing Vertex AI Gemini client...")
        client = create_vertex_client(args.project, args.region, system_prompt, examples)
        logger.info("Client initialized")

        # Determine how many to process
        total = len(df) if args.limit == 0 else min(args.limit, len(df))

        # Resume from checkpoint if available
        start_row = 0
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        errors = 0

        if checkpoint and args.resume:
            start_row = checkpoint['processed_rows']
            results = checkpoint['results']
            total_input_tokens = checkpoint['stats'].get('input_tokens', 0)
            total_output_tokens = checkpoint['stats'].get('output_tokens', 0)
            errors = checkpoint['stats'].get('errors', 0)
            logger.info(f"Resuming from row {start_row}")

        logger.info(f"Will process {total - start_row} transcripts with {args.workers} concurrent workers")

        start_time = time.time()

        for i in range(start_row, total, args.batch_size):
            batch_end = min(i + args.batch_size, total)
            batch_df = df.iloc[i:batch_end]
            batch_num = (i - start_row) // args.batch_size + 1
            total_batches = ((total - start_row) + args.batch_size - 1) // args.batch_size

            logger.info(f"Batch {batch_num}/{total_batches} (rows {i}-{batch_end-1})...")

            # Process batch concurrently
            batch_results = process_batch_concurrent(
                client,
                batch_df['body'].tolist(),
                system_prompt,
                examples,
                MODEL,
                i,
                args.workers,
                logger
            )

            # Merge results with original rows — explode multi-problem rows
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                result = batch_results[idx]

                total_input_tokens += result.get('_input_tokens', 0)
                total_output_tokens += result.get('_output_tokens', 0)

                if result.get('_error'):
                    errors += 1
                    logger.warning(f"Row {result.get('_row_idx')}: {result.get('_error')}")

                for problem in result.get('problems', []):
                    merged = row.to_dict()
                    merged['summarised_problem'] = problem.get('summarised_problem', 'ERROR')
                    merged['fidelity'] = problem.get('fidelity', 'unknown')
                    merged['journey'] = problem.get('journey', 'unknown')
                    merged['stage'] = problem.get('stage', 'unknown')
                    merged['mechanism'] = problem.get('mechanism', 'unknown')
                    merged['failure_mode'] = problem.get('failure_mode', 'unknown')
                    merged['team'] = problem.get('team', 'unknown')
                    merged['problem_type'] = problem.get('problem_type', 'unknown')
                    merged['impact'] = problem.get('impact', 'unknown')
                    merged['cause_fidelity'] = problem.get('cause_fidelity', 'N/A')
                    results.append(merged)

            # Save checkpoint after each batch
            save_checkpoint(checkpoint_path, batch_end, results, {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'errors': errors
            })
            logger.info(f"  Checkpoint saved at row {batch_end}")

            # Progress update
            processed = batch_end
            elapsed = time.time() - start_time
            rows_done = processed - start_row
            rate = rows_done / elapsed if elapsed > 0 else 0
            remaining = total - processed
            eta = remaining / rate if rate > 0 else 0
            logger.info(f"  Progress: {processed}/{total} ({rate:.1f}/sec, ETA: {eta:.0f}s)")

            if processed % 100 == 0 or processed == total:
                tracker.checkpoint(f"Processed {processed} rows", processed)

        # Create output DataFrame
        result_df = pd.DataFrame(results)

        # Save as Parquet
        logger.info(f"Saving to {args.output}...")
        result_df.to_parquet(args.output, index=False)

        # Delete checkpoint after successful completion
        delete_checkpoint(checkpoint_path)
        logger.info("Checkpoint cleared (job complete)")

        # Summary
        tracker.complete(args.output, len(result_df))

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {MODEL}")
        logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        logger.info(f"Throughput: {(total - start_row)/elapsed:.1f} transcripts/sec")
        logger.info(f"\nToken usage:")
        logger.info(f"  Input tokens: {total_input_tokens:,}")
        logger.info(f"  Output tokens: {total_output_tokens:,}")

        # Estimate costs (Gemini 2.0 Flash pricing)
        input_cost = (total_input_tokens / 1_000_000) * 0.10
        output_cost = (total_output_tokens / 1_000_000) * 0.40
        total_cost = input_cost + output_cost

        logger.info(f"\nEstimated cost (Gemini 2.0 Flash pricing):")
        logger.info(f"  Input: ${input_cost:.4f}")
        logger.info(f"  Output: ${output_cost:.4f}")
        logger.info(f"  TOTAL: ${total_cost:.4f}")

        if errors > 0:
            logger.warning(f"\nErrors: {errors} rows had issues (check logs)")
        logger.info(f"\nNext step: python3 embed.py {args.output} data/embedded.parquet")

    except Exception as e:
        tracker.fail(e)
        logger.error(f"\nJob failed! Checkpoint saved at: {checkpoint_path}")
        logger.error(f"To resume, run with --resume flag")
        raise


if __name__ == '__main__':
    main()
