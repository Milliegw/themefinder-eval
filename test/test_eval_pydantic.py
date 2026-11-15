import json

import pytest
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel.span_tree import SpanTree

from eval_pydantic import ThemeF1, build_prediction_lookup


def test_theme_f1_expected_a_b_predicted_a():
    evaluator = ThemeF1(theme_names=["A", "B", "C"])
    ctx = EvaluatorContext(
        name="case-1",
        inputs="Example comment",
        metadata={"theme_order": ["A", "B", "C"]},
        expected_output={"A", "B"},
        output={"A"},
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    assert evaluator.evaluate_sync(ctx) == pytest.approx(2 / 3)


def test_build_prediction_lookup_uses_response_id(tmp_path):
    run_json = tmp_path / "themefinder_test.json"
    run_json.write_text(
        json.dumps(
            {
                "result": {
                    "themes": [{"topic_id": "A", "topic": "Education and support services"}],
                    "mapping": [
                        {"response_id": 7, "response": "Example response", "labels": ["A"]},
                    ],
                }
            }
        )
    )

    lookup = build_prediction_lookup(run_json)

    assert 7 in lookup
    assert lookup[7] == {"Education and skills"}
