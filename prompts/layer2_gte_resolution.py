"""
Layer 2: GTE Relation Candidate Resolution

Deterministic semantic retrieval - NOT an LLM layer.
Resolves short relation candidate phrases to actual KG relations via cosine similarity.
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors."""
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


def resolve_relation_candidates(
    relation_candidates: List[str],
    sample_relation_list: List[str],
    gte_model: SentenceTransformer,
    top_k: int = 5,
    similarity_threshold: float = 0.3
) -> Dict[str, List[Dict[str, float]]]:
    """
    Resolve relation candidate phrases to actual KG relations via GTE retrieval.

    Args:
        relation_candidates: Short phrases from Layer 1 (e.g., ["department to country"])
        sample_relation_list: Available KG relations for this sample
        gte_model: Initialized GTE embedding model
        top_k: Number of relations to retrieve per candidate
        similarity_threshold: Minimum cosine similarity to include a result

    Returns:
        Dict mapping each candidate phrase to list of {relation, score} tuples,
        sorted by similarity (highest first).
    """
    if not relation_candidates or not sample_relation_list:
        return {}

    # Encode both sets
    candidate_embeddings = gte_model.encode(relation_candidates, normalize_embeddings=True)
    relation_embeddings = gte_model.encode(sample_relation_list, normalize_embeddings=True)

    # Compute similarity matrix
    similarities = cosine_similarity(candidate_embeddings, relation_embeddings)

    # For each candidate, retrieve top-k
    resolved = {}
    for i, candidate in enumerate(relation_candidates):
        candidate_sims = similarities[i]

        # Get top-k indices
        top_k_actual = min(top_k, len(sample_relation_list))
        top_indices = np.argsort(candidate_sims)[-top_k_actual:][::-1]

        # Filter by threshold and build results
        results = []
        for idx in top_indices:
            score = float(candidate_sims[idx])
            if score >= similarity_threshold:
                results.append({
                    "relation": sample_relation_list[idx],
                    "score": score
                })

        resolved[candidate] = results

    return resolved


def resolve_for_intent_chain(
    intent_chain: List[Dict],
    sample_relation_list: List[str],
    gte_model: SentenceTransformer
) -> List[Dict]:
    """
    Resolve relations for an entire intent chain from Layer 1.

    Args:
        intent_chain: List of {step, intent, relation_candidates} from Layer 1
        sample_relation_list: Available KG relations for this sample
        gte_model: Initialized GTE embedding model

    Returns:
        Enhanced intent chain with resolved_relations added to each step.
    """
    enhanced_chain = []

    for step in intent_chain:
        candidates = step.get("relation_candidates", [])
        resolved = resolve_relation_candidates(
            candidates,
            sample_relation_list,
            gte_model
        )

        # Flatten resolved relations for this step (deduplicated)
        all_relations = set()
        for results in resolved.values():
            for r in results:
                all_relations.add(r["relation"])

        enhanced_step = step.copy()
        enhanced_step["resolved_relations"] = list(all_relations)
        enhanced_step["resolved_details"] = resolved  # Keep scores for debugging
        enhanced_chain.append(enhanced_step)

    return enhanced_chain


# =============================================================================
# Integration Example
# =============================================================================

if __name__ == "__main__":
    # Example usage
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Layer 1 output
    layer1_output = {
        "anchor": "Alta Verapaz Department",
        "intent_chain": [
            {
                "step": 1,
                "intent": "find what nation contains this department",
                "relation_candidates": ["department to country", "administrative division country", "state province country"]
            },
            {
                "step": 2,
                "intent": "verify the nation is in Central America",
                "relation_candidates": ["country continent", "country location", "country region"]
            }
        ],
        "target_entity": "Central America"
    }

    # Sample's available relations (from CWQ dataset)
    sample_relations = [
        "location.administrative_division.country",
        "location.country.first_level_divisions",
        "location.location.contains",
        "location.country.continent",
        "location.country.location",
        "government.governmental_jurisdiction.border_with_country"
    ]

    # Resolve
    enhanced = resolve_for_intent_chain(
        layer1_output["intent_chain"],
        sample_relations,
        model
    )

    for step in enhanced:
        print(f"Step {step['step']}: {step['intent']}")
        print(f"  Resolved: {step['resolved_relations']}")
        print()
