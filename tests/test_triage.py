import numpy as np
import pandas as pd

from secom_project.triage import EVIDENCE_COLUMNS, build_consensus_ranking, make_run_level_triage


def test_candidate_ranking_is_repeatable_and_missing_safe():
    evidence = pd.DataFrame({"feature": ["b", "a", "c"], "effect_size": [1.0, -2.0, np.nan]})
    for index, column in enumerate(EVIDENCE_COLUMNS):
        evidence[column] = [index, np.nan, index + 1]
    first = build_consensus_ranking(evidence)
    second = build_consensus_ranking(evidence)
    pd.testing.assert_frame_equal(first, second)
    assert not first.isna().any().any()


def test_run_level_triage_has_one_row_per_run():
    scores = pd.DataFrame(
        {
            "run_id": [0, 1, 2],
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "failure": [0, 1, 0],
            "phase": ["phase_i", "phase_ii", "final_holdout"],
            "model_risk": [0.1, 0.8, 0.4],
            "spc_score": [0.2, 1.4, 0.9],
            "t2": [1.0, 2.0, 3.0],
            "q_spe": [0.5, 0.7, 0.6],
            "ewma_alert": [False, True, False],
            "cusum_alert": [False, True, False],
            "mspc_alert": [False, False, True],
        }
    )
    contributions = pd.DataFrame({"run_id": [0, 1, 2], "x": [1.0, 0.0, 2.0], "y": [0.0, 2.0, 1.0]})
    result = make_run_level_triage(scores, contributions)
    assert len(result) == len(scores)
    assert result["run_id"].is_unique


def test_seeded_random_process_is_fixed():
    assert np.array_equal(np.random.default_rng(42).random(10), np.random.default_rng(42).random(10))
