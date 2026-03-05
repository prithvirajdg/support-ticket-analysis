"""
Compare two or more cluster output CSVs/Parquets side by side.

Metrics reported per file:
  - Number of clusters
  - Noise % (cluster == -1)
  - Cluster size distribution (min, median, max, top-5 sizes)
  - Purity: for each cluster, what fraction of rows share the dominant value
    across journey, team, and failure_mode. Higher = more homogeneous clusters.

Usage:
    python3 compare_clusters.py data/clustered_baseline.csv data/clustered_team_fm.csv
    python3 compare_clusters.py data/clustered_baseline.csv data/clustered_team_fm.csv data/clustered_fm_mech.csv
"""

import sys
import os
import argparse


PURITY_COLS = ['journey', 'team', 'failure_mode', 'fidelity']


def load(path):
    import pandas as pd
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def cluster_purity(df, col):
    """
    For each non-noise cluster, compute the fraction of rows that share the
    dominant value of `col`. Returns the mean purity across all clusters.
    1.0 = perfectly homogeneous. Lower = mixed.
    """
    if col not in df.columns:
        return None
    clustered = df[df['cluster'] != -1]
    if clustered.empty:
        return None
    purities = (
        clustered.groupby('cluster')[col]
        .apply(lambda s: s.value_counts().iloc[0] / len(s))
    )
    return purities.mean()


def summarise(path):
    import numpy as np
    df = load(path)
    label = os.path.basename(path)

    if 'cluster' not in df.columns:
        print(f"  ERROR: no 'cluster' column in {path}")
        return

    total = len(df)
    n_noise = int((df['cluster'] == -1).sum())
    noise_pct = 100 * n_noise / total

    clustered = df[df['cluster'] != -1]
    n_clusters = clustered['cluster'].nunique()

    sizes = clustered.groupby('cluster').size()

    print(f"\n{'='*60}")
    print(f"FILE:     {label}")
    print(f"{'='*60}")
    print(f"Total rows:     {total}")
    print(f"Clusters:       {n_clusters}")
    print(f"Noise:          {n_noise}  ({noise_pct:.1f}%)")

    if not sizes.empty:
        print(f"\nCluster size distribution:")
        print(f"  Min:    {sizes.min()}")
        print(f"  Median: {int(sizes.median())}")
        print(f"  Mean:   {sizes.mean():.1f}")
        print(f"  Max:    {sizes.max()}")
        top5 = sizes.sort_values(ascending=False).head(5)
        print(f"  Top 5:  {list(top5.values)}")

    print(f"\nPurity (mean fraction of dominant value per cluster):")
    for col in PURITY_COLS:
        p = cluster_purity(df, col)
        if p is not None:
            bar = '#' * int(p * 20)
            print(f"  {col:<15} {p:.3f}  |{bar:<20}|")
        else:
            print(f"  {col:<15} (column not present)")

    if 'stratum' in df.columns:
        n_strata = df['stratum'].nunique()
        print(f"\nStratification: {n_strata} strata")
        strat_sizes = df.groupby('stratum').size().sort_values(ascending=False)
        for s, n in strat_sizes.head(5).items():
            print(f"  {s}: {n} rows")
        if len(strat_sizes) > 5:
            print(f"  ... and {len(strat_sizes) - 5} more")
    else:
        print(f"\nStratification: none (global clustering)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare cluster output files side by side'
    )
    parser.add_argument('files', nargs='+', help='CSV/Parquet cluster output files to compare')
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Provide at least 2 files to compare.")
        sys.exit(1)

    for path in args.files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    for path in args.files:
        summarise(path)

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'File':<35} {'Clusters':>8} {'Noise%':>7}  Purity (journey / team / failure_mode)")
    print('-' * 90)

    import pandas as pd
    for path in args.files:
        df = load(path)
        label = os.path.basename(path)[:34]
        clustered = df[df['cluster'] != -1]
        n_clusters = clustered['cluster'].nunique() if not clustered.empty else 0
        noise_pct = 100 * (df['cluster'] == -1).sum() / len(df)

        purities = []
        for col in ['journey', 'team', 'failure_mode']:
            p = cluster_purity(df, col)
            purities.append(f"{p:.2f}" if p is not None else " N/A")

        print(f"{label:<35} {n_clusters:>8} {noise_pct:>6.1f}%  {' / '.join(purities)}")

    print()


if __name__ == '__main__':
    main()
