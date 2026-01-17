from typing import Dict, List


def hit_rate_at_k(
    relevant_doc_ids: List[str],
    retrieved_doc_ids: List[str],
    k: int,
) -> float:
    hits = 0
    for rel_id in relevant_doc_ids:
        if rel_id in retrieved_doc_ids[:k]:
            hits += 1
    if not relevant_doc_ids:
        return 0.0
    return hits / len(relevant_doc_ids)


def mrr_at_k(
    relevant_doc_ids: List[str],
    retrieved_doc_ids: List[str],
    k: int,
) -> float:
    rr_sum = 0.0
    for rel_id in relevant_doc_ids:
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            if doc_id == rel_id:
                rr = 1.0 / rank
                break
        rr_sum += rr
    if not relevant_doc_ids:
        return 0.0
    return rr_sum / len(relevant_doc_ids)


def evaluate_retrieval_batch(
    batch_relevant: List[List[str]],
    batch_retrieved: List[List[str]],
    k: int = 5,
) -> Dict[str, float]:
    assert len(batch_relevant) == len(batch_retrieved)
    hit_rates = []
    mrrs = []
    for relevant, retrieved in zip(batch_relevant, batch_retrieved):
        hit_rates.append(hit_rate_at_k(relevant, retrieved, k))
        mrrs.append(mrr_at_k(relevant, retrieved, k))
    hit_rate_mean = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    mrr_mean = sum(mrrs) / len(mrrs) if mrrs else 0.0
    return {"hit_rate": hit_rate_mean, "mrr": mrr_mean}

