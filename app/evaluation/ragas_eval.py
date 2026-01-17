from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevance,
    context_precision,
    faithfulness,
)


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

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance, context_precision],
    )

    scores = result.to_pandas().mean().to_dict()
    return {
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "answer_relevance": float(scores.get("answer_relevance", 0.0)),
        "context_precision": float(scores.get("context_precision", 0.0)),
    }

