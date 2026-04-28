"""Bridge loads screenshots per scenario when with_screenshots=True."""
from unittest.mock import patch, MagicMock


def test_prepare_bulk_entries_loads_screenshots_per_scenario():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0",
        conv_id="conv_xyz",
        cut_turn=10,
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello",
        student_context="ctx",
        last_student_message="hello",
        mode="detected",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0",
        tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )
    fake_screenshots = [
        {"filename": "s1.jpg", "anchor_turn": 6, "storage_path": "deidentified/screenshots/REAL/s1.jpg", "timestamp_seconds": 6.0},
    ]

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots",
               return_value=fake_screenshots) as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
            with_screenshots=True,
        )

    # load_anchored_screenshots called with original conv_id, not scenario_id
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == "conv_xyz"

    # build_analysis_entries got screenshots_by_conv keyed by scenario_id
    kwargs = mock_build.call_args.kwargs
    sbc = kwargs.get("screenshots_by_conv")
    assert sbc == {"conv_xyz__det_0": fake_screenshots}
    assert kwargs.get("with_screenshots") is True


def test_prepare_bulk_entries_default_no_screenshots():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0", conv_id="conv_xyz", cut_turn=10,
        transcript_prefix="Turn 1. TUTOR: hi", student_context="ctx",
        last_student_message="hi", mode="detected",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0", tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots") as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
        )

    mock_load.assert_not_called()
    assert mock_build.call_args.kwargs.get("screenshots_by_conv") is None
    assert mock_build.call_args.kwargs.get("with_screenshots") is False
