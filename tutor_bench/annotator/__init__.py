"""Annotator: detect key moments, annotate action/result, and label effectiveness.

Pilot (Plan 016) exposes Pass 1 only. Pass 2 and Pass 3 follow in separate
plans; their functions will be added to ``__all__`` as they land.
"""

from tutor_bench.annotator.config import (
    AnnotatorConfig,
    ModelConfig,
    RetryConfig,
    load_annotator_config,
)
from tutor_bench.annotator.detect_key_moments import (
    KeyMoment,
    ParseError,
    PhaseResult,
    detect_key_moments,
)

__all__ = [
    "AnnotatorConfig",
    "KeyMoment",
    "ModelConfig",
    "ParseError",
    "PhaseResult",
    "RetryConfig",
    "detect_key_moments",
    "load_annotator_config",
]
