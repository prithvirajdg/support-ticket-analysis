"""
Embed summarised user problems using sentence-transformers.
Input: Parquet/CSV with 'summarised_user_problem' column (from disaggregate.py)
Output: Parquet with original data + embedding vectors

Usage:
    python3 embed.py data/disaggregated.parquet data/embedded.parquet
"""

import argparse
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Embed summarised user problems')
    parser.add_argument('input', help='Input Parquet/CSV with summarised_user_problem column')
    parser.add_argument('output', help='Output Parquet file with embeddings')
    parser.add_argument('--batch-size', type=int, default=256, help='Embedding batch size (default: 256)')
    args = parser.parse_args()

    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from pipeline_utils import (
        setup_logging, validate_columns, StepTracker, ValidationError
    )

    logger = setup_logging("embed")
    tracker = StepTracker("EMBEDDING", logger)

    try:
        tracker.start("Generating embeddings with sentence-transformers")

        # Validate input file exists
        if not os.path.exists(args.input):
            raise ValidationError(f"Input file not found: {args.input}")

        tracker.checkpoint("Input validated")

        # Load model
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded (384 dimensions)")
        tracker.checkpoint("Model loaded")

        # Load data
        logger.info(f"Loading {args.input}...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")

        # Validate required column
        validate_columns(df, ['summarised_user_problem'], "embed", logger)
        tracker.checkpoint("Data loaded", len(df))

        # Filter out NA and error summaries (no actionable problem)
        valid_mask = (
            df['summarised_user_problem'].notna() &
            (df['summarised_user_problem'] != 'NA') &
            (df['summarised_user_problem'] != 'PARSE_ERROR') &
            (df['summarised_user_problem'] != 'API_ERROR') &
            (df['summarised_user_problem'] != 'ERROR')
        )
        valid_df = df[valid_mask].copy()
        skipped = len(df) - len(valid_df)

        if skipped > 0:
            logger.warning(f"Skipping {skipped} rows with NA/errors")
            # Log a sample of skipped rows
            skipped_df = df[~valid_mask]
            if len(skipped_df) > 0:
                logger.debug(f"Sample skipped values: {skipped_df['summarised_user_problem'].head(5).tolist()}")

        if len(valid_df) == 0:
            raise ValidationError(
                "No valid rows to embed! All rows have NA or ERROR in summarised_user_problem. "
                "Check disaggregation step for issues."
            )

        logger.info(f"Embedding {len(valid_df)} valid summaries...")
        tracker.checkpoint("Filtered valid rows", len(valid_df))

        # Generate embeddings with progress
        summaries = valid_df['summarised_user_problem'].tolist()
        logger.info(f"Processing in batches of {args.batch_size}...")

        embeddings = model.encode(
            summaries,
            show_progress_bar=True,
            batch_size=args.batch_size
        )

        logger.info(f"Created embeddings: {embeddings.shape}")
        tracker.checkpoint("Embeddings generated", len(embeddings))

        # Add embeddings as columns (embedding_0, embedding_1, ..., embedding_383)
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=valid_df.index)

        result_df = pd.concat([valid_df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)

        # Save as Parquet (much smaller than CSV for embeddings)
        logger.info(f"Saving to {args.output}...")
        result_df.to_parquet(args.output, index=False)

        # Report size comparison
        output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Output size: {output_size_mb:.1f} MB")

        tracker.complete(args.output, len(result_df))

        logger.info(f"\nRows: {len(result_df)} embedded, {skipped} skipped")
        logger.info(f"Dimensions: {embeddings.shape[1]}")
        logger.info(f"\nNext step: python3 -m modal run cluster_modal.py --input {args.output} --output data/clustered.parquet")

    except Exception as e:
        tracker.fail(e)
        raise


if __name__ == '__main__':
    main()
