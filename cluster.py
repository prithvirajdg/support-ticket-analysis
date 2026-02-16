"""
Cluster embedded summaries using UMAP + HDBSCAN.
Runs locally on Compute Engine VM.

UMAP reduces dimensions for better clustering.
HDBSCAN discovers natural clusters without specifying K.

Usage:
    python3 cluster.py data/embedded.parquet data/clustered.parquet

    # Adjust parameters
    python3 cluster.py data/embedded.parquet data/clustered.parquet --umap-dims 15 --min-cluster-size 20

Scaling notes:
    - 20k rows: Works fine on e2-standard-4 (4 vCPU, 16GB RAM)
    - 100k rows: Use e2-standard-8 (8 vCPU, 32GB RAM)
    - 500k rows: Use e2-highmem-8 (8 vCPU, 64GB RAM) or process in batches
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Cluster embedded summaries using UMAP + HDBSCAN'
    )
    parser.add_argument('input', help='Input Parquet/CSV with embedding columns')
    parser.add_argument('output', help='Output Parquet file with cluster assignments')
    parser.add_argument('--umap-dims', type=int, default=15,
                        help='Target dimensions for UMAP reduction (default: 15)')
    parser.add_argument('--umap-neighbors', type=int, default=30,
                        help='UMAP n_neighbors parameter (default: 30)')
    parser.add_argument('--min-cluster-size', type=int, default=15,
                        help='HDBSCAN minimum cluster size (default: 15)')
    parser.add_argument('--show-samples', type=int, default=5,
                        help='Number of samples to show per cluster (default: 5)')
    args = parser.parse_args()

    import pandas as pd
    import numpy as np
    from pipeline_utils import (
        setup_logging, validate_columns, StepTracker, ValidationError
    )

    logger = setup_logging("cluster")
    tracker = StepTracker("CLUSTERING (UMAP + HDBSCAN)", logger)

    try:
        tracker.start(f"Clustering with UMAP ({args.umap_dims} dims) + HDBSCAN (min_size={args.min_cluster_size})")

        # Validate input
        if not os.path.exists(args.input):
            raise ValidationError(f"Input file not found: {args.input}")

        tracker.checkpoint("Input validated")

        # Load data
        logger.info(f"Loading {args.input}...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")

        # Extract embedding columns
        embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
        if not embedding_cols:
            raise ValidationError(
                "No embedding columns found! Expected columns like 'embedding_0', 'embedding_1', etc. "
                "Make sure the embed step completed successfully."
            )

        logger.info(f"Found {len(embedding_cols)} embedding dimensions")
        tracker.checkpoint("Data loaded", len(df))

        # Extract embeddings
        embeddings = df[embedding_cols].values.astype(np.float32)
        logger.info(f"Embedding matrix shape: {embeddings.shape}")
        logger.info(f"Memory usage: {embeddings.nbytes / (1024**2):.1f} MB")

        # Step 1: UMAP dimensionality reduction
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: UMAP Dimensionality Reduction")
        logger.info(f"{'='*60}")
        logger.info(f"Reducing {embeddings.shape[1]} dims → {args.umap_dims} dims")
        logger.info(f"Parameters: n_neighbors={args.umap_neighbors}, min_dist=0.0, metric=cosine")

        import umap

        start_umap = time.time()
        reducer = umap.UMAP(
            n_components=args.umap_dims,
            n_neighbors=args.umap_neighbors,
            min_dist=0.0,           # Tighter clusters for clustering task
            metric='cosine',        # Good for text embeddings
            random_state=42,        # Reproducibility
            low_memory=True,        # Better for large datasets
            n_jobs=-1               # Use all CPUs
        )
        reduced = reducer.fit_transform(embeddings)
        umap_time = time.time() - start_umap

        logger.info(f"UMAP complete: {umap_time:.1f}s")
        logger.info(f"Reduced shape: {reduced.shape}")
        tracker.checkpoint("UMAP complete", reduced.shape[1])

        # Step 2: HDBSCAN clustering
        logger.info(f"\n{'='*60}")
        logger.info("STEP 2: HDBSCAN Clustering")
        logger.info(f"{'='*60}")
        logger.info(f"Parameters: min_cluster_size={args.min_cluster_size}, metric=euclidean")

        import hdbscan

        start_hdbscan = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass - better for varying density
            core_dist_n_jobs=-1              # Use all CPUs
        )
        cluster_labels = clusterer.fit_predict(reduced)
        hdbscan_time = time.time() - start_hdbscan

        logger.info(f"HDBSCAN complete: {hdbscan_time:.1f}s")
        tracker.checkpoint("HDBSCAN complete")

        # Calculate stats
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()

        logger.info(f"\nClusters found: {n_clusters}")
        logger.info(f"Noise points: {n_noise} ({100*n_noise/len(df):.1f}%)")

        # Add cluster labels to dataframe
        df['cluster'] = cluster_labels

        # Remove embedding columns from output (save space)
        output_cols = [c for c in df.columns if not c.startswith('embedding_')]
        output_df = df[output_cols]

        # Save results (CSV for final output, Parquet if specified)
        logger.info(f"\nSaving to {args.output}...")
        if args.output.endswith('.csv'):
            output_df.to_csv(args.output, index=False)
        else:
            output_df.to_parquet(args.output, index=False)

        output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Output size: {output_size_mb:.1f} MB")

        tracker.complete(args.output, len(output_df))

        # Print cluster summary
        logger.info(f"\n{'='*70}")
        logger.info("CLUSTER SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Clusters found: {n_clusters}")
        logger.info(f"Noise points: {n_noise} ({100*n_noise/len(df):.1f}%)")
        logger.info(f"UMAP time: {umap_time:.1f}s")
        logger.info(f"HDBSCAN time: {hdbscan_time:.1f}s")
        logger.info(f"Total time: {umap_time + hdbscan_time:.1f}s")

        # Sort clusters by size
        cluster_sizes = df[df['cluster'] != -1].groupby('cluster').size().sort_values(ascending=False)

        for cluster_id in cluster_sizes.index[:10]:  # Top 10 clusters
            cluster_df = df[df['cluster'] == cluster_id]
            size = len(cluster_df)

            # Get domain breakdown
            if 'domain' in cluster_df.columns:
                domains = cluster_df['domain'].value_counts().head(2)
                domain_str = ", ".join([f"{d}: {c}" for d, c in domains.items()])
            else:
                domain_str = "N/A"

            # Get impact breakdown
            if 'impact_type' in cluster_df.columns:
                impacts = cluster_df['impact_type'].value_counts().head(2)
                impact_str = ", ".join([f"{i}: {c}" for i, c in impacts.items()])
            else:
                impact_str = "N/A"

            logger.info(f"\n--- Cluster {cluster_id} ({size} items) ---")
            logger.info(f"    Domains: {domain_str}")
            logger.info(f"    Impact: {impact_str}")
            logger.info(f"    Sample problems:")

            if 'summarised_user_problem' in cluster_df.columns:
                samples = cluster_df['summarised_user_problem'].head(args.show_samples).tolist()
                for sample in samples:
                    display = str(sample)[:75] + "..." if len(str(sample)) > 75 else sample
                    logger.info(f"      - {display}")

        if n_clusters > 10:
            logger.info(f"\n... and {n_clusters - 10} more clusters")

        # Noise samples
        if n_noise > 0:
            logger.info(f"\n--- Noise/Outliers ({n_noise} items) ---")
            noise_df = df[df['cluster'] == -1]
            if 'summarised_user_problem' in noise_df.columns:
                samples = noise_df['summarised_user_problem'].head(args.show_samples).tolist()
                for sample in samples:
                    display = str(sample)[:75] + "..." if len(str(sample)) > 75 else sample
                    logger.info(f"      - {display}")

        logger.info(f"\n{'='*70}")
        logger.info(f"Output: {args.output}")
        logger.info("Tip: Filter by cluster column, review samples, assign a problem name.")
        logger.info(f"{'='*70}")

    except Exception as e:
        tracker.fail(e)
        raise


if __name__ == '__main__':
    main()
