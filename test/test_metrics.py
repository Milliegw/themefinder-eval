import pytest, random
from src.metrics import precision, recall, f1
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1

# 1 = theme present, 0 = theme absent
y_true = [1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 1, 0, 1, 0]


def test_precision():
    # 3 true positives, 1 false positive
    assert precision(y_true, y_pred) == pytest.approx(0.75)


def test_recall():
    # 3 true positives, 1 false negative
    assert recall(y_true, y_pred) == pytest.approx(0.75)


def test_f1():
    # 3 true positives, 1 false positive, 1 false negative
    assert f1(y_true, y_pred) == pytest.approx(0.75)


def test_no_predicted_positives():
    # model never predicts the theme: precision is undefined, return 0.0
    assert precision([1, 0, 1], [0, 0, 0]) == 0.0


def test_lengths_must_match():
    with pytest.raises(ValueError):
        precision([1, 0], [1])

def test_matches_sklearn():
    random.seed(42)
    y_true = [random.randint(0, 1) for _ in range(100)]
    y_pred = [random.randint(0, 1) for _ in range(100)]

    assert precision(y_true, y_pred) == pytest.approx(precision_score(y_true, y_pred))
    assert recall(y_true, y_pred) == pytest.approx(recall_score(y_true, y_pred))
    assert f1(y_true, y_pred) == pytest.approx(sk_f1(y_true, y_pred))

def test_precision_and_recall_differ():
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    # TP=2, FP=1, FN=2
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(0.5)