"""
Multilingual Review Summarizer — Claude Haiku Batch API
Handles: English, Hindi (Devanagari), Hinglish (romanized), and code-mixed text
Output:  English summary per review, with distinct problems split into separate rows

Features:
- Prompt caching on system prompt (massive input cost reduction)
- Batch API (50% off vs real-time API)
- Resume support (--resume) to recover from partial runs
- Splits negative/mixed reviews into one row per distinct problem
- Positive reviews kept as a single row

Usage:
    pip install anthropic pandas tqdm
    export ANTHROPIC_API_KEY=your_key_here

    # Fresh run:
    python batch_summarize.py --input reviews.csv --text-col content

    # Resume a partial run using saved batch IDs:
    python batch_summarize.py --input reviews.csv --text-col content --resume batch_ids.json
"""

from __future__ import annotations
import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300           # more room for JSON array of problems
BATCH_SIZE = 100           # requests per batch job (Anthropic allows up to 10K)
POLL_INTERVAL = 30         # seconds between status checks
BATCH_IDS_FILE = "batch_ids.json"   # saved automatically for resume support
MIN_WORD_COUNT = 5                  # skip reviews shorter than this

SYSTEM_PROMPT_FILE = "system_prompt_summarize.txt"
EXAMPLES_FILE = "examples_summarize.txt"


def load_prompt_blocks() -> list[dict]:
    """Load system prompt and examples from text files, with prompt caching."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    system_path = os.path.join(script_dir, SYSTEM_PROMPT_FILE)
    examples_path = os.path.join(script_dir, EXAMPLES_FILE)

    with open(system_path) as f:
        system_text = f.read().strip()
    with open(examples_path) as f:
        examples_text = f.read().strip()

    return [
        {
            "type": "text",
            "text": system_text + "\n\n" + examples_text,
            "cache_control": {"type": "ephemeral"}
        }
    ]

USER_PROMPT_TEMPLATE = "App: {app_type}\nReview: {text}\nOutput:"

# ── Core Functions ─────────────────────────────────────────────────────────────

def load_reviews(filepath: str, text_col: str) -> pd.DataFrame:
    """Load CSV and validate the text column exists."""
    df = pd.read_csv(filepath)
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )
    print(f"Loaded {len(df):,} rows from {filepath}")
    return df


def build_batch_requests(texts: list[str], start_idx: int, system_blocks: list[dict], app_type: str) -> list[dict]:
    """Convert a list of review texts into Anthropic batch request format."""
    requests = []
    for i, text in enumerate(texts):
        clean_text = str(text).strip() if pd.notna(text) else ""
        if not clean_text:
            clean_text = "[empty]"

        requests.append({
            "custom_id": str(start_idx + i),
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": system_blocks,
                "messages": [
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(app_type=app_type, text=clean_text)
                    }
                ]
            }
        })
    return requests


def submit_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    """Submit a batch job and return its ID."""
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    """Poll until the batch job is complete."""
    print(f"  Waiting for batch {batch_id}...", end="", flush=True)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        if status == "ended":
            counts = batch.request_counts
            print(f"\n  Done — succeeded: {counts.succeeded}, errored: {counts.errored}")
            return
        elif status in ("canceling", "canceled"):
            raise RuntimeError(f"Batch {batch_id} was canceled.")

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def collect_results(client: anthropic.Anthropic, batch_id: str) -> dict[int, str]:
    """Stream results from a completed batch, return {row_index: raw_text}."""
    results = {}
    for result in client.messages.batches.results(batch_id):
        row_idx = int(result.custom_id)

        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text.strip()
        elif result.result.type == "errored":
            error_msg = result.result.error.error.message
            raw = json.dumps([{"sentiment": "neutral", "summary": f"[ERROR: {error_msg}]"}])
        else:
            raw = json.dumps([{"sentiment": "neutral", "summary": "[SKIPPED]"}])

        results[row_idx] = raw
    return results


def parse_and_explode(df: pd.DataFrame, raw_results: dict[int, str]) -> pd.DataFrame:
    """Parse JSON results and explode into one row per distinct problem."""
    exploded_rows = []

    for row_idx, raw_json in raw_results.items():
        original_row = df.iloc[row_idx]

        try:
            # Handle cases where model wraps JSON in markdown code blocks
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            items = [{"sentiment": "unknown", "summary": f"[PARSE ERROR] {raw_json[:200]}"}]

        for problem_num, item in enumerate(items, start=1):
            row = original_row.to_dict()
            row["sentiment"] = item.get("sentiment", "unknown")
            row["emotion"] = item.get("emotion", "unknown")
            row["body"] = item.get("summary", "[MISSING]")
            row["problem_number"] = problem_num
            row["total_problems"] = len(items)
            exploded_rows.append(row)

    return pd.DataFrame(exploded_rows)


def save_batch_ids(batch_ids: list[str], path: str):
    """Persist submitted batch IDs so a failed run can be resumed."""
    with open(path, "w") as f:
        json.dump({"batch_ids": batch_ids, "saved_at": time.time()}, f, indent=2)
    print(f"Batch IDs saved to {path} (use --resume to recover if interrupted)")


def load_batch_ids(path: str) -> list[str]:
    """Load previously saved batch IDs for a resume run."""
    with open(path) as f:
        data = json.load(f)
    batch_ids = data["batch_ids"]
    print(f"Resuming {len(batch_ids)} batch job(s) from {path}")
    return batch_ids


# ── Main Orchestration ─────────────────────────────────────────────────────────

def run(input_file: str, text_col: str, app_type: str, output_file: str, resume_file: str | None):
    # Load API key from ~/.env if not already in environment
    if "ANTHROPIC_API_KEY" not in os.environ:
        env_path = os.path.expanduser("~/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # 1. Load data and prompt
    df = load_reviews(input_file, text_col)
    system_blocks = load_prompt_blocks()
    print(f"Loaded prompt from {SYSTEM_PROMPT_FILE} + {EXAMPLES_FILE}")

    # Filter out short reviews
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    original_count = len(df)
    df = df[df["word_count"] >= MIN_WORD_COUNT].reset_index(drop=True)
    print(f"Filtered: {original_count:,} -> {len(df):,} reviews ({MIN_WORD_COUNT}+ words)")

    texts = df[text_col].tolist()
    total = len(texts)
    all_results: dict[int, str] = {}

    # 2. Either resume existing batches or submit new ones
    if resume_file:
        batch_ids = load_batch_ids(resume_file)
    else:
        chunks = [texts[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
        print(f"\nSubmitting {len(chunks)} batch(es) of up to {BATCH_SIZE} requests each...")
        print("   (System prompt caching active — cache hits apply after first request)\n")

        batch_ids = []
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Submitting")):
            start_idx = chunk_num * BATCH_SIZE
            requests = build_batch_requests(chunk, start_idx, system_blocks, app_type)
            batch_id = submit_batch(client, requests)
            batch_ids.append(batch_id)

        save_batch_ids(batch_ids, BATCH_IDS_FILE)

    # 3. Wait for each batch and collect results
    print(f"\nWaiting for {len(batch_ids)} batch job(s) to complete...")
    for batch_id in batch_ids:
        wait_for_batch(client, batch_id)
        results = collect_results(client, batch_id)
        all_results.update(results)

    # 4. Parse JSON and explode into one row per problem
    result_df = parse_and_explode(df, all_results)

    # Flag any rows that didn't get a result
    processed_indices = set(all_results.keys())
    missing_indices = set(range(total)) - processed_indices
    if missing_indices:
        print(f"Warning: {len(missing_indices)} rows missing results — marked for rerun")
        for idx in missing_indices:
            row = df.iloc[idx].to_dict()
            row["sentiment"] = "unknown"
            row["emotion"] = "unknown"
            row["body"] = "[MISSING - rerun]"
            row["problem_number"] = 1
            row["total_problems"] = 1
            result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

    # 5. Save output
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    neg_mixed = result_df[result_df["sentiment"].isin(["negative", "mixed"])]
    print(f"\nDone! Output: {output_file}")
    print(f"  Total input reviews:  {total:,}")
    print(f"  Total output rows:    {len(result_df):,}")
    print(f"  Negative/mixed rows:  {len(neg_mixed):,}")
    print(f"  Positive rows:        {len(result_df) - len(neg_mixed):,}")

    # Clean up batch IDs file on successful completion
    if os.path.exists(BATCH_IDS_FILE):
        os.remove(BATCH_IDS_FILE)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize multilingual reviews via Claude Batch API")
    parser.add_argument("--input",    required=True,  help="Path to input CSV file")
    parser.add_argument("--text-col", required=True,  help="Column name containing review text")
    parser.add_argument("--app-type", required=True,  choices=["partner", "customer"],
                        help="Type of app: 'partner' or 'customer'")
    parser.add_argument("--output",   default=None,   help="Output CSV path (default: input_summarized.csv)")
    parser.add_argument("--resume",   default=None,   metavar="BATCH_IDS_JSON",
                        help="Path to batch_ids.json from a previous interrupted run")
    args = parser.parse_args()

    output_file = args.output or args.input.replace(".csv", "_summarized.csv")
    run(args.input, args.text_col, args.app_type, output_file, args.resume)
