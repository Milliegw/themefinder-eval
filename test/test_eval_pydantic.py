import pytest
from pydantic_evals.evaluators import EvaluatorContext

from eval_pydantic import ThemeF1


def test_theme_f1_expected_a_b_predicted_a():
    evaluator = ThemeF1(theme_names=["A", "B", "C"])
    ctx = EvaluatorContext(
        name="case-1",
        inputs="Example comment",
        metadata={"theme_order": ["A", "B", "C"]},
        expected_output={"A", "B"},
        output={"A"},
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    assert evaluator.evaluate_sync(ctx) == pytest.approx(2 / 3)
