from __future__ import annotations

from typing import Any

DEFAULT_RETRY_STATUSES = [429, 500, 502, 503, 504]
DEFAULT_COMPLEXITY = {
    "low_max_chars": 1200,
    "medium_max_chars": 6000,
    "high_max_chars": 16000,
}
DEFAULT_CLASSIFIER_CALIBRATION: dict[str, Any] = {
    "enabled": False,
    "min_samples": 30,
    "target_secondary_success_rate": 0.8,
    "secondary_low_confidence_min_confidence": 0.18,
    "secondary_mixed_signal_min_confidence": 0.35,
    "adjustment_step": 0.03,
    "deadband": 0.05,
    "min_threshold": 0.05,
    "max_threshold": 0.9,
}
DEFAULT_ROUTE_RERANKER: dict[str, Any] = {
    "enabled": False,
    "backend": "local_embedding",
    "local_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "local_files_only": True,
    "local_max_length": 256,
    "similarity_weight": 0.35,
    "min_similarity": 0.0,
    "model_hints": {},
}
DEFAULT_LEARNED_ROUTING: dict[str, Any] = {
    "enabled": False,
    "bias": -4.0,
    "default_output_tokens": 512,
    "feature_weights": {},
    "task_candidates": {},
    "utility_weights": {
        "cost": 12.0,
        "latency": 0.2,
        "failure": 3.0,
    },
}
