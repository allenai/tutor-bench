# Data Directory

Private data — gitignored. Do not commit files from this directory.

For transparency, code that reproduces the numbers in this README.md can be found in data/stats.ipynb. 

---

## transcripts/step_up.jsonl

Deidentified tutoring session transcripts from StepUp. Each line is one session.

**21,445 records across 4 batches** (`2025-10-16`, `2026-02-13`, `2026-03-24`, `2026-04-27`).

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `transcript_id` | string (UUID) | Unique identifier for this session |
| `source` | string | Always `"step_up"` |
| `batch` | string (date) | Ingestion batch (YYYY-MM-DD) |
| `has_video` | bool | Whether video was available during transcription |
| `turns` | array | Ordered list of conversation turns (see below) |
| `session` | object | Session metadata (see below) |
| `eedi_content_link` | array | Links to Eedi math content shown during session (often empty) |
| `enrichments` | array | Non-speech events interleaved with turns (see below) |

### `turns` items

| Field | Type | Description |
|---|---|---|
| `turn_number` | int | 1-indexed turn position |
| `start_seconds` | int | Turn start time in seconds from session start |
| `end_seconds` | int | Turn end time in seconds |
| `role` | string | `"Tutor"` or `"Student"` |
| `text` | string | Transcribed speech |

Average ~349 turns per session.

### `session` fields

| Field | Type | Description |
|---|---|---|
| `transcription_by` | string | Model/service used to transcribe (e.g. `"SUT (Gemini 2.5 Pro)"`) |
| `primary_language` | string | Session language (may be empty) |
| `prior_session_count` | int | Number of prior sessions this tutor–student pair had |
| `session_id` | string (UUID) | Platform session ID |
| `tutor_id` | string (UUID) | Deidentified tutor ID |
| `student_id` | string (UUID) | Deidentified student ID |

### `enrichments` items

Non-speech events (pauses, screen interactions, etc.) that occurred between or before turns.

| Field | Type | Description |
|---|---|---|
| `type` | string | Event category (e.g. `"pause"`, `"screen_interaction"`) |
| `start_seconds` | int | Event start time |
| `end_seconds` | int | Event end time |
| `before_turn` | int | This event occurred before this turn number |
| `label` | string | Short human-readable label |
| `content` | string | Description of what happened |

---

## teacher_annotations/step_up_annotations.jsonl

Human annotations on Step Up tutoring sessions, produced through a structured annotation interface. Each line is one annotation record for one session. A session may have multiple records (one per annotation type).

**1,095 records** — all from batch `2025-10-16`. Three annotation types: `caption` (79), `scaffolding` (311), `rapport` (705).

**Note:** 0.6% of rapport and 0.9% of scaffolding `turn_annotations` have `null` for both `turn_number_start` and `turn_number_end`. These are annotator entries that have substantive SAR content but were not anchored to specific turns. `build_ground_truth.py` skips these.

### Top-level fields (all types)

| Field | Type | Description |
|---|---|---|
| `transcript_id` | string (UUID) | Links to a record in `step_up.jsonl` |
| `source` | string | Always `"step_up"` |
| `batch` | string (date) | Annotation batch |
| `annotation_type` | string | `"caption"`, `"scaffolding"`, or `"rapport"` |
| `interface_version` | string | Annotation interface version (may be `"untracked"`) |
| `annotator_id` | string | Deidentified annotator identifier |
| `turn_annotations` | array | Segment-level annotations (see per-type detail below) |

### `caption` annotations

Free-text commentary on transcript segments. Also includes overall session-level annotations.

**`turn_annotations` items:**

| Field | Type | Description |
|---|---|---|
| `turn_number_start` | int | First turn of the annotated segment |
| `turn_number_end` | int | Last turn of the annotated segment |
| `annotation_timestamp` | string (ISO 8601) | When this annotation was submitted |
| `text` | string | Free-text commentary on the segment |

**Additional top-level field — `overall_annotations`** (object):

| Field | Description |
|---|---|
| `tutor_effectiveness` | Overall assessment of the tutor's effectiveness in the session |
| `student_learning_indicators` | Observed signals of whether the student learned |
| `student_behavior` | Notable student behaviors the annotator attended to |
| `student_description` | Description of the student's engagement and learning style |

### `scaffolding` and `rapport` annotations

Structured SAR (Situation–Action–Result) annotations on specific moments in the session.

**`turn_annotations` items:**

| Field | Type | Description |
|---|---|---|
| `turn_number_start` | int | First turn of the annotated moment |
| `turn_number_end` | int | Last turn of the annotated moment |
| `annotation_timestamp` | string (ISO 8601) | When this annotation was submitted |
| `situation` | string | What was happening in the session at this moment |
| `action` | string | What the tutor did (the pedagogical move) |
| `result` | string | How effective the move was and what the outcome was |
