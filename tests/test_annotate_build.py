"""Tests for annotator.core.annotate entry building with screenshots."""
import pytest


class TestBuildAnalysisEntriesWithScreenshots:
    def _detections_for(self, conv_id):
        return {
            conv_id: {
                "detections": [
                    {"turn_start": 1, "turn_end": 1, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 1"},
                    {"turn_start": 3, "turn_end": 3, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 3"},
                ],
                "usage": {},
            }
        }

    def test_per_moment_image_filtering(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        # Fixture: turns at start_seconds 0/3/10, screenshot at 4.000s -> anchors to turn 2.
        # Moments are at turn 1 and turn 3.
        # context_window=0 -> windows [1,1] and [3,3]: neither includes turn 2 -> no images.
        # context_window=1 -> windows [1,2] and [2,3]: both include turn 2 -> image attached.
        entries_zero = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=0, version="v4",
            with_screenshots=True,
        )
        for e in entries_zero:
            assert "images" not in e["request"]

        entries_one = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=1, version="v4",
            with_screenshots=True,
        )
        for e in entries_one:
            assert e["request"]["images"] == [
                "deidentified/screenshots/099bf759-abcd/4.000.jpg"
            ]

    def test_no_images_when_flag_off(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        entries = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=20, version="v4",
            with_screenshots=False,
        )
        for e in entries:
            assert "images" not in e["request"]
