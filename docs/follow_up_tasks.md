# Follow-up Tasks

Open items surfaced during other work. Each entry: what, why, and where to start.

## `Evaluator._load_jsonlines` duplicates `io_utils.load_jsonl`

**What:** [tutor_bench/evaluator.py:153](../tutor_bench/evaluator.py#L153) defines `_load_jsonlines(filepath: str)`, a private JSONL loader that predates the shared `load_jsonl` in [tutor_bench/toolkit/io_utils.py](../tutor_bench/toolkit/io_utils.py).

**Why it matters:** Two loaders means two places to fix bugs and two places where storage-backend support (now `STORAGE_ROOT` / `UPath` aware) has to be re-implemented. The evaluator's loader does not benefit from S3 routing.

**Where to start:** Replace the body (or call sites) of `_load_jsonlines` with `tutor_bench.toolkit.io_utils.load_jsonl`. Verify [tests/](../tests/) coverage exercises the evaluator's load path before/after.

## `test_load_with_api` is skipped after Plan 016 mock removal

**What:** [tests/test_grade_school_fixtures.py:104](../tests/test_grade_school_fixtures.py#L104) (`test_load_with_api`) is `@pytest.mark.skip`'d. It exercised the mock `Annotator(...).process_transcripts(...)` API plus the mock `Evaluator`. Plan 016 removed the mock Annotator and replaces it with `detect_key_moments(...) -> PhaseResult`, which has different semantics.

**Why it matters:** The test was the only end-to-end smoke test for the public API. While skipped, there is no coverage of `from tutor_bench import ...` working for a typical user flow.

**Where to start:** Rewrite the test once (a) `detect_key_moments` is wired up at the end of Plan 016, and (b) the Evaluator is migrated in a future plan. New version should: call `detect_key_moments` against the grade-school fixture with a stubbed `ModelClient` and assert it returns a non-empty `PhaseResult`. Drop the Evaluator portion until that side of the migration lands.
