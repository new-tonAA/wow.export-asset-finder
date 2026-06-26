"""
WoW Asset Finder - Step 2: Search for similar models given an input image.

Usage:
    python search.py <input_image_path> [--top-k 10]

Loads the pre-computed feature vectors from wow_model_features.npz,
extracts a CLIP vector from the input image, and returns the top-K
most similar WoW model paths.
"""

import sys
import argparse
import io
from pathlib import Path

import numpy as np
import faiss
from PIL import Image

import torch
import open_clip


def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load CLIP model for feature extraction."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess, device


def extract_image_features(image_path, model, preprocess, device):
    """Extract CLIP feature vector from an input image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten().astype(np.float32)


def build_faiss_index(vectors):
    """Build a FAISS index from feature vectors (cosine similarity via inner product)."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim for normalized vectors
    index.add(vectors)
    return index


def search(query_vector, index, paths, top_k=10):
    """Search for the most similar models."""
    query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        results.append({
            "rank": i + 1,
            "path": str(paths[idx]),
            "similarity": float(dist),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Search WoW models by image similarity")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument(
        "--features",
        type=str,
        default=str(Path(__file__).parent / "wow_model_features.npz"),
        help="Path to the feature vectors file",
    )
    args = parser.parse_args()

    # Validate input
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not Path(args.features).exists():
        print(f"Error: Features file not found: {args.features}")
        print("Run extract_features.py first to generate feature vectors.")
        sys.exit(1)

    # Load features
    print("Loading feature vectors...")
    data = np.load(args.features, allow_pickle=True)
    paths = data["paths"]
    vectors = data["vectors"].astype(np.float32)
    print(f"Loaded {len(paths)} model features ({vectors.shape[1]}D vectors).")

    # Build FAISS index
    print("Building search index...")
    index = build_faiss_index(vectors)

    # Load CLIP and extract query features
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()

    print(f"Extracting features from: {args.image}")
    query_vector = extract_image_features(args.image, model, preprocess, device)

    # Search
    results = search(query_vector, index, paths, top_k=args.top_k)

    # Display results
    print(f"\n{'='*60}")
    print(f"Top {args.top_k} similar WoW models:")
    print(f"{'='*60}")
    for r in results:
        print(f"  #{r['rank']:2d}  [sim: {r['similarity']:.4f}]  {r['path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
