"""tutor-bench: A benchmarking and evaluation framework for AI tutoring systems."""

from tutor_bench.annotator import (
    AnnotatorConfig,
    KeyMoment,
    PhaseResult,
    detect_key_moments,
    load_annotator_config,
)
from tutor_bench.evaluator import Evaluator
from tutor_bench.version import __version__

__all__ = [
    "AnnotatorConfig",
    "Evaluator",
    "KeyMoment",
    "PhaseResult",
    "__version__",
    "detect_key_moments",
    "load_annotator_config",
]
