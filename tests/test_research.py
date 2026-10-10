import json
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from training.research_analysis import research_metrics, shap_summary


def test_threshold_counts_and_net_benefit():
    result = research_metrics([0, 0, 1, 1], [.1, .6, .4, .9])
    row = result['threshold_analysis'][49]
    assert row['threshold'] == .5
    assert [row[k] for k in ['TN', 'FP', 'FN', 'TP']] == [1, 1, 1, 1]
    assert row['net_benefit'] == 0
    assert row['sensitivity'] == .5
    assert row['specificity'] == .5
    assert result['evaluation_positive'] == 2
    assert len(result['precision_recall_curve']['recall']) == 5
    json.dumps(result, allow_nan=False)


def test_perfect_predictions():
    result = research_metrics([0, 1], [0, 1])
    assert result['brier_score'] == 0
    assert all(row['net_benefit'] == .5 for row in result['threshold_analysis'])


def test_shap_summary_aggregates_only():
    X = np.random.default_rng(1).normal(size=(30, 2))
    model = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, X[:, 0] > 0)
    result = shap_summary(model, X, ['A', 'B'], sample_size=10)
    assert result['sample_size'] == 10
    assert set(result['mean_absolute']) == {'A', 'B'}
    for feature in ['A', 'B']:
        assert sum(p['Count'] for p in result['points'] if p['Feature'] == feature) == 10
    assert all(set(p) == {'Feature', 'SHAP', 'Value', 'Count'} for p in result['points'])
    json.dumps(result, allow_nan=False)
