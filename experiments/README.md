# Experiments

This directory holds research-quality variants that should not ship with the library:

- Prompt iteration (multiple versions of the same prompt under test)
- Per-archetype / per-style prompt variants used by ad-hoc analyses
- One-off scoring scripts and exploratory notebooks

Library code in `tutor_bench/` always uses one canonical prompt per function. When iteration is needed, copy the canonical prompt here, vary it, and run experiments outside the library quality gate.
