from typing import Dict, List

from datasets import Dataset
from ragas import evaluate

try:
    from ragas.metrics import faithfulness
except Exception:
    faithfulness = None

try:
    from ragas.metrics import answer_relevance
except Exception:
    answer_relevance = None

try:
    from ragas.metrics import context_precision
except Exception:
    context_precision = None


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> Dict[str, float]:
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    metrics = [
        m
        for m in [faithfulness, answer_relevance, context_precision]
        if m is not None
    ]

    if not metrics:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
        }

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    scores = result.to_pandas().mean().to_dict()
    return {
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "answer_relevance": float(scores.get("answer_relevance", 0.0)),
        "context_precision": float(scores.get("context_precision", 0.0)),
    }
