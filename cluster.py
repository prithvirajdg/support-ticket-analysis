"""
Cluster embedded summaries using UMAP + HDBSCAN.
Runs locally on Compute Engine VM.

UMAP reduces dimensions for better clustering.
HDBSCAN discovers natural clusters without specifying K.

Usage:
    # Baseline — global clustering (original behaviour, no stratification)
    python3 cluster.py data/embedded.parquet data/clustered_baseline.csv

    # Stratified by team × failure_mode
    python3 cluster.py data/embedded.parquet data/clustered_team_fm.csv --stratify-by team failure_mode

    # Stratified by failure_mode × mechanism
    python3 cluster.py data/embedded.parquet data/clustered_fm_mech.csv --stratify-by failure_mode mechanism

    # Stratified by all three
    python3 cluster.py data/embedded.parquet data/clustered_all3.csv --stratify-by team failure_mode mechanism

    # Compare all runs
    python3 compare_clusters.py data/clustered_baseline.csv data/clustered_team_fm.csv data/clustered_fm_mech.csv

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


def run_umap_hdbscan(embeddings, umap_dims, umap_neighbors, min_cluster_size, prob_threshold):
    """
    Run UMAP + HDBSCAN on a matrix of embeddings.
    Caps umap_dims and umap_neighbors automatically when the stratum is small.
    Returns (cluster_labels, probabilities, umap_time, hdbscan_time).
    """
    import numpy as np
    import umap as umap_lib
    import hdbscan as hdbscan_lib

    n = len(embeddings)
    actual_dims = min(umap_dims, n - 2)
    actual_neighbors = min(umap_neighbors, n - 1)

    start_umap = time.time()
    reducer = umap_lib.UMAP(
        n_components=actual_dims,
        n_neighbors=actual_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        low_memory=True,
        n_jobs=-1
    )
    reduced = reducer.fit_transform(embeddings)
    umap_time = time.time() - start_umap

    start_hdbscan = time.time()
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(reduced)
    hdbscan_time = time.time() - start_hdbscan

    if prob_threshold > 0.0:
        labels[clusterer.probabilities_ < prob_threshold] = -1

    return labels, clusterer.probabilities_, umap_time, hdbscan_time


def main():
    parser = argparse.ArgumentParser(
        description='Cluster embedded summaries using UMAP + HDBSCAN'
    )
    parser.add_argument('input', help='Input Parquet/CSV with embedding columns')
    parser.add_argument('output', help='Output CSV/Parquet file with cluster assignments')
    parser.add_argument('--umap-dims', type=int, default=15,
                        help='Target dimensions for UMAP reduction (default: 15)')
    parser.add_argument('--umap-neighbors', type=int, default=30,
                        help='UMAP n_neighbors parameter (default: 30)')
    parser.add_argument('--min-cluster-size', type=int, default=15,
                        help='HDBSCAN minimum cluster size (default: 15)')
    parser.add_argument('--show-samples', type=int, default=5,
                        help='Number of samples to show per cluster (default: 5)')
    parser.add_argument('--prob-threshold', type=float, default=0.0,
                        help='Minimum HDBSCAN membership probability (default: 0.0 = keep all)')
    parser.add_argument('--stratify-by', nargs='+', default=None, metavar='COL',
                        help=(
                            'Run UMAP+HDBSCAN independently per stratum defined by these columns, '
                            'then merge with globally unique cluster IDs. Adds a "stratum" column to output. '
                            'Examples: --stratify-by team failure_mode | '
                            '--stratify-by failure_mode mechanism | '
                            '--stratify-by team failure_mode mechanism'
                        ))
    args = parser.parse_args()

    import pandas as pd
    import numpy as np
    from pipeline_utils import (
        setup_logging, validate_columns, StepTracker, ValidationError
    )

    logger = setup_logging("cluster")
    tracker = StepTracker("CLUSTERING (UMAP + HDBSCAN)", logger)

    try:
        strat_label = (
            f" | stratified by: {' × '.join(args.stratify_by)}"
            if args.stratify_by else " | global (no stratification)"
        )
        tracker.start(
            f"UMAP ({args.umap_dims} dims) + HDBSCAN (min_size={args.min_cluster_size}){strat_label}"
        )

        if not os.path.exists(args.input):
            raise ValidationError(f"Input file not found: {args.input}")

        tracker.checkpoint("Input validated")

        logger.info(f"Loading {args.input}...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")

        embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
        if not embedding_cols:
            raise ValidationError(
                "No embedding columns found. Expected columns like 'embedding_0', 'embedding_1', etc. "
                "Make sure the embed step completed successfully."
            )

        logger.info(f"Found {len(embedding_cols)} embedding dimensions")
        tracker.checkpoint("Data loaded", len(df))

        # Reset index so positional indexing aligns with the embeddings array
        df = df.reset_index(drop=True)
        embeddings = df[embedding_cols].values.astype(np.float32)
        logger.info(f"Embedding matrix shape: {embeddings.shape}")
        logger.info(f"Memory usage: {embeddings.nbytes / (1024**2):.1f} MB")

        # -------------------------------------------------------------------
        # STRATIFIED MODE
        # -------------------------------------------------------------------
        if args.stratify_by:
            missing = [c for c in args.stratify_by if c not in df.columns]
            if missing:
                raise ValidationError(
                    f"Stratify columns not found in data: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

            logger.info(f"\n{'='*60}")
            logger.info(f"STRATIFIED CLUSTERING: {' × '.join(args.stratify_by)}")
            logger.info(f"{'='*60}")

            strat_groups = list(df.groupby(args.stratify_by, sort=True))
            logger.info(f"Strata found: {len(strat_groups)}")

            all_labels = np.full(len(df), -1, dtype=int)
            stratum_col = pd.Series([''] * len(df), dtype=str)
            cluster_offset = 0
            total_umap_time = 0.0
            total_hdbscan_time = 0.0

            for keys, group in strat_groups:
                stratum_name = (
                    ' / '.join(str(k) for k in keys)
                    if isinstance(keys, tuple) else str(keys)
                )
                pos_idx = group.index.tolist()
                n = len(pos_idx)
                stratum_col.iloc[pos_idx] = stratum_name

                # Skip strata too small for HDBSCAN to produce meaningful clusters
                min_required = args.min_cluster_size * 2
                if n < min_required:
                    logger.info(
                        f"  {stratum_name}: {n} rows — below threshold ({min_required}), all noise"
                    )
                    continue

                stratum_embeddings = embeddings[pos_idx]
                labels, _, umap_t, hdbscan_t = run_umap_hdbscan(
                    stratum_embeddings,
                    args.umap_dims, args.umap_neighbors,
                    args.min_cluster_size, args.prob_threshold
                )
                total_umap_time += umap_t
                total_hdbscan_time += hdbscan_t

                # Map local cluster IDs → globally unique IDs
                max_local = int(labels.max()) if (labels >= 0).any() else -1
                global_labels = np.where(labels >= 0, labels + cluster_offset, -1)
                if max_local >= 0:
                    cluster_offset += max_local + 1

                for i, pos in enumerate(pos_idx):
                    all_labels[pos] = global_labels[i]

                n_local = len(set(labels[labels >= 0]))
                n_noise = int((labels == -1).sum())
                logger.info(
                    f"  {stratum_name}: {n} rows → {n_local} clusters, "
                    f"{n_noise} noise ({100*n_noise/n:.1f}%)"
                )

            df['cluster'] = all_labels
            df['stratum'] = stratum_col
            umap_time = total_umap_time
            hdbscan_time = total_hdbscan_time

        # -------------------------------------------------------------------
        # GLOBAL MODE — original behaviour, unchanged
        # -------------------------------------------------------------------
        else:
            logger.info(f"\n{'='*60}")
            logger.info("STEP 1: UMAP Dimensionality Reduction")
            logger.info(f"{'='*60}")
            logger.info(f"Reducing {embeddings.shape[1]} dims → {args.umap_dims} dims")
            logger.info(f"Parameters: n_neighbors={args.umap_neighbors}, min_dist=0.0, metric=cosine")

            logger.info(f"\n{'='*60}")
            logger.info("STEP 2: HDBSCAN Clustering")
            logger.info(f"{'='*60}")
            logger.info(f"Parameters: min_cluster_size={args.min_cluster_size}, metric=euclidean")

            cluster_labels, probs, umap_time, hdbscan_time = run_umap_hdbscan(
                embeddings,
                args.umap_dims, args.umap_neighbors,
                args.min_cluster_size, args.prob_threshold
            )

            logger.info(f"UMAP complete: {umap_time:.1f}s")
            logger.info(f"HDBSCAN complete: {hdbscan_time:.1f}s")

            if args.prob_threshold > 0.0:
                n_demoted = (probs < args.prob_threshold).sum()
                logger.info(
                    f"Probability threshold {args.prob_threshold}: demoted {n_demoted} points to noise"
                )

            df['cluster'] = cluster_labels

        # -------------------------------------------------------------------
        # SHARED: stats, save, summary
        # -------------------------------------------------------------------
        tracker.checkpoint("Clustering complete")

        n_clusters = len(set(df['cluster'].values[df['cluster'].values >= 0]))
        n_noise = int((df['cluster'] == -1).sum())

        logger.info(f"\nClusters found: {n_clusters}")
        logger.info(f"Noise points:   {n_noise} ({100*n_noise/len(df):.1f}%)")

        # Remove embedding columns from output (save space)
        output_cols = [c for c in df.columns if not c.startswith('embedding_')]
        output_df = df[output_cols]

        logger.info(f"\nSaving to {args.output}...")
        if args.output.endswith('.csv'):
            output_df.to_csv(args.output, index=False)
        else:
            output_df.to_parquet(args.output, index=False)

        output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info(f"Output size: {output_size_mb:.1f} MB")

        tracker.complete(args.output, len(output_df))

        # Print top-10 clusters by size
        logger.info(f"\n{'='*70}")
        logger.info("CLUSTER SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total rows:     {len(df)}")
        logger.info(f"Clusters found: {n_clusters}")
        logger.info(f"Noise points:   {n_noise} ({100*n_noise/len(df):.1f}%)")
        logger.info(f"UMAP time:      {umap_time:.1f}s")
        logger.info(f"HDBSCAN time:   {hdbscan_time:.1f}s")

        cluster_sizes = (
            df[df['cluster'] != -1]
            .groupby('cluster').size()
            .sort_values(ascending=False)
        )

        for cluster_id in cluster_sizes.index[:10]:
            cluster_df = df[df['cluster'] == cluster_id]
            size = len(cluster_df)

            journey_str = "N/A"
            if 'journey' in cluster_df.columns:
                journeys = cluster_df['journey'].value_counts().head(2)
                journey_str = ", ".join([f"{j}: {c}" for j, c in journeys.items()])

            team_str = "N/A"
            if 'team' in cluster_df.columns:
                teams = cluster_df['team'].value_counts().head(2)
                team_str = ", ".join([f"{t}: {c}" for t, c in teams.items()])

            fidelity_str = "N/A"
            if 'fidelity' in cluster_df.columns:
                fidelities = cluster_df['fidelity'].value_counts()
                fidelity_str = ", ".join([f"{f}: {c}" for f, c in fidelities.items()])

            strat_str = ""
            if 'stratum' in cluster_df.columns:
                strat_str = f"\n    Stratum:  {cluster_df['stratum'].iloc[0]}"

            logger.info(f"\n--- Cluster {cluster_id} ({size} items) ---")
            logger.info(f"    Journey:  {journey_str}")
            logger.info(f"    Team:     {team_str}")
            logger.info(f"    Fidelity: {fidelity_str}{strat_str}")
            logger.info(f"    Sample problems:")

            if 'summarised_problem' in cluster_df.columns:
                for sample in cluster_df['summarised_problem'].head(args.show_samples):
                    display = str(sample)[:75] + "..." if len(str(sample)) > 75 else sample
                    logger.info(f"      - {display}")

        if n_clusters > 10:
            logger.info(f"\n... and {n_clusters - 10} more clusters")

        if n_noise > 0:
            logger.info(f"\n--- Noise/Outliers ({n_noise} items) ---")
            noise_df = df[df['cluster'] == -1]
            if 'summarised_problem' in noise_df.columns:
                for sample in noise_df['summarised_problem'].head(args.show_samples):
                    display = str(sample)[:75] + "..." if len(str(sample)) > 75 else sample
                    logger.info(f"      - {display}")

        logger.info(f"\n{'='*70}")
        logger.info(f"Output: {args.output}")
        if args.stratify_by:
            logger.info(f"Stratified by: {' × '.join(args.stratify_by)}")
            logger.info("Compare with other runs: python3 compare_clusters.py <file1> <file2> ...")
        else:
            logger.info("Tip: Run with --stratify-by to compare stratified clustering.")
            logger.info("     python3 compare_clusters.py <baseline.csv> <stratified.csv>")
        logger.info(f"{'='*70}")

    except Exception as e:
        tracker.fail(e)
        raise


if __name__ == '__main__':
    main()
