"""
Test that clustering works by meaning, not words.
Uses scikit-learn (more compatible) - same principle as FAISS.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

# These use DIFFERENT WORDS but describe the SAME underlying problems
core_problems = [
    # Cluster 1: Self-troubleshooting failures (VPN)
    "users aren't able to troubleshoot issues themselves when they ran into VPN connectivity issues",
    "user couldn't resolve VPN problems on their own without contacting support",
    "self-service troubleshooting failed for network connectivity issues",

    # Cluster 2: Billing discrepancies
    "there are discrepancies in billing amounts",
    "user was charged incorrectly and billing doesn't match expected amount",
    "payment amount doesn't match what was quoted",

    # Cluster 3: Need step-by-step guidance
    "user wants step-by-step guidance on how they can integrate analytics tools",
    "user needs detailed instructions for setting up integrations",
    "looking for documentation on how to configure third-party tools"
]

print("Embedding problems...")
embeddings = model.encode(core_problems)
print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}\n")

# Cluster using sklearn KMeans
print("Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(embeddings)

# Show results
print("=" * 60)
print("CLUSTERING RESULTS")
print("=" * 60)
print("\nThese problems use DIFFERENT WORDS but should cluster by MEANING:\n")

for cluster_id in range(3):
    print(f"--- Cluster {cluster_id} ---")
    for i, problem in enumerate(core_problems):
        if cluster_assignments[i] == cluster_id:
            print(f"  • {problem[:70]}...")
    print()

print("=" * 60)
print("SUCCESS: If each cluster contains related problems, clustering works!")
print("=" * 60)
