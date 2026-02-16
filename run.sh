#!/bin/bash

# =============================================================================
# SUPPORT TICKET CLUSTERING PIPELINE (GCP / VERTEX AI VERSION)
# =============================================================================
# Designed for 20K-500K tickets with:
# - Gemini 2.0 Flash via Vertex AI
# - Local sentence-transformers for embeddings
# - UMAP + HDBSCAN for clustering
# - Parquet format (5-10x smaller than CSV)
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# GCP Settings
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="us-central1"  # Vertex AI Gemini region (widely available)

# Input: Raw transcripts (supports glob patterns for multiple files)
INPUT_PATTERN="data/*.csv"

# Intermediate files (Parquet format)
DISAGGREGATED="data/disaggregated.parquet"
EMBEDDED="data/embedded.parquet"

# Output: Final clustered results
OUTPUT="data/clustered.parquet"

# Processing options
LIMIT=0                  # 0 = process all, or set a number for testing
WORKERS=10               # Concurrent Claude API calls

# Clustering options
MIN_CLUSTER_SIZE=15      # Minimum items to form a cluster
UMAP_DIMS=15             # Dimensions after UMAP reduction
UMAP_NEIGHBORS=30        # UMAP n_neighbors parameter

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_status() {
    echo "[$(date '+%H:%M:%S')] $1"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Expected output file not found: $1"
        echo "Check the logs directory for detailed error messages."
        exit 1
    fi
    local size=$(du -h "$1" | cut -f1)
    print_status "Output created: $1 ($size)"
}

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

print_header "PRE-FLIGHT CHECKS"

# Check GCP project is set
if [ "$GOOGLE_CLOUD_PROJECT" = "your-project-id" ]; then
    echo "ERROR: GOOGLE_CLOUD_PROJECT not set."
    echo "Either:"
    echo "  1. Edit this script and set GOOGLE_CLOUD_PROJECT"
    echo "  2. Or run: export GOOGLE_CLOUD_PROJECT=your-actual-project-id"
    exit 1
fi
print_status "GCP Project: $GOOGLE_CLOUD_PROJECT"
print_status "Region: $REGION"

# Check for input files
INPUT_COUNT=$(ls -1 $INPUT_PATTERN 2>/dev/null | wc -l | tr -d ' ')
if [ "$INPUT_COUNT" -eq 0 ]; then
    echo "ERROR: No input files found matching: $INPUT_PATTERN"
    echo "Expected CSV files in the data/ directory."
    exit 1
fi
print_status "Found $INPUT_COUNT input file(s) matching: $INPUT_PATTERN"

# Check gcloud is authenticated
if ! gcloud auth application-default print-access-token &> /dev/null; then
    echo "ERROR: Not authenticated with GCP."
    echo "Run: gcloud auth application-default login"
    exit 1
fi
print_status "GCP authentication valid"

# Check Python packages
python3 -c "import vertexai, pandas, sentence_transformers, umap, hdbscan" 2>/dev/null || {
    echo "ERROR: Required Python packages not found."
    echo "Install with:"
    echo "  pip install google-cloud-aiplatform pandas sentence-transformers umap-learn hdbscan pyarrow"
    exit 1
}
print_status "Python packages available"

# Create logs directory
mkdir -p logs
print_status "Logs directory ready"

# =============================================================================
# PIPELINE
# =============================================================================

print_header "PIPELINE CONFIGURATION"
echo "GCP Project:       $GOOGLE_CLOUD_PROJECT"
echo "Region:            $REGION"
echo "Input pattern:     $INPUT_PATTERN"
echo "Output:            $OUTPUT"
echo "Limit:             $LIMIT (0 = all)"
echo "Workers:           $WORKERS concurrent API calls"
echo "Min cluster size:  $MIN_CLUSTER_SIZE"
echo "UMAP dimensions:   $UMAP_DIMS"
echo ""

# Step 1: Disaggregate
print_header "STEP 1/3: DISAGGREGATION (Gemini via Vertex AI)"
print_status "Starting disaggregation..."

python3 disaggregate.py \
    --input "$INPUT_PATTERN" \
    --output "$DISAGGREGATED" \
    --project "$GOOGLE_CLOUD_PROJECT" \
    --region "$REGION" \
    --workers "$WORKERS" \
    --limit "$LIMIT"

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "PIPELINE FAILED AT: DISAGGREGATION"
    echo "============================================================"
    echo "Check logs/disaggregate_*.log for details"
    exit 1
fi

check_file "$DISAGGREGATED"

# Step 2: Embed
print_header "STEP 2/3: EMBEDDING (sentence-transformers)"
print_status "Starting embedding..."

python3 embed.py "$DISAGGREGATED" "$EMBEDDED"

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "PIPELINE FAILED AT: EMBEDDING"
    echo "============================================================"
    echo "Check logs/embed_*.log for details"
    exit 1
fi

check_file "$EMBEDDED"

# Step 3: Cluster
print_header "STEP 3/3: CLUSTERING (UMAP + HDBSCAN)"
print_status "Starting clustering..."

python3 cluster.py \
    "$EMBEDDED" \
    "$OUTPUT" \
    --umap-dims "$UMAP_DIMS" \
    --umap-neighbors "$UMAP_NEIGHBORS" \
    --min-cluster-size "$MIN_CLUSTER_SIZE"

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "PIPELINE FAILED AT: CLUSTERING"
    echo "============================================================"
    echo "Check logs/cluster_*.log for details"
    exit 1
fi

check_file "$OUTPUT"

# =============================================================================
# SUCCESS
# =============================================================================

print_header "PIPELINE COMPLETE"
echo ""
echo "Output:     $OUTPUT"
echo ""
echo "Files created:"
echo "  - $DISAGGREGATED (disaggregated transcripts)"
echo "  - $EMBEDDED (with embeddings)"
echo "  - $OUTPUT (with cluster assignments)"
echo ""
echo "Logs:       logs/*.log"
echo ""
echo "Next steps:"
echo "  1. Open $OUTPUT in a data tool (Python, Excel, etc.)"
echo "  2. Filter by cluster column"
echo "  3. Review samples in each cluster"
echo "  4. Assign descriptive names to clusters"
echo ""

# Quick stats
echo "Quick stats:"
python3 -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT')
n_clusters = df[df['cluster'] != -1]['cluster'].nunique()
n_noise = (df['cluster'] == -1).sum()
print(f'  Total rows:   {len(df)}')
print(f'  Clusters:     {n_clusters}')
print(f'  Noise/outliers: {n_noise} ({100*n_noise/len(df):.1f}%)')
"
