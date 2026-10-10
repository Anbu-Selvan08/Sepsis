"""Offline, record-level research summaries. No patient identifiers are exported."""
import numpy as np
from sklearn.metrics import precision_recall_curve, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve


def research_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    precision, recall, _ = precision_recall_curve(y, probability)
    observed, predicted = calibration_curve(y, probability, n_bins=10)
    rows = []
    prevalence = float(y.mean())
    for threshold in np.linspace(.01, .99, 99):
        tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
        weight = threshold / (1 - threshold)
        benefit = (tp - fp * weight) / len(y)
        all_benefit = prevalence - (1 - prevalence) * weight
        rows.append(dict(threshold=float(threshold), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                         sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                         precision=float(tp / (tp + fp)) if tp + fp else 0.,
                         net_benefit=float(benefit), treat_all=float(all_benefit), treat_none=0.,
                         advantage=float(benefit - max(0, all_benefit))))
    return dict(precision_recall_curve=dict(precision=precision.tolist(), recall=recall.tolist()),
                calibration=dict(predicted=predicted.tolist(), observed=observed.tolist()),
                brier_score=float(brier_score_loss(y, probability)), threshold_analysis=rows,
                evaluation_records=len(y), evaluation_positive=int(y.sum()))


def shap_summary(model, X, features, sample_size=100):
    rng = np.random.default_rng(42)
    sample = np.asarray(X)[rng.choice(len(X), min(sample_size, len(X)), replace=False)]
    try:
        import shap
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        raw_values = explainer.shap_values(sample)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[list(model.classes_).index(1)])
        elif getattr(raw_values, 'ndim', 0) == 3:
            values = np.asarray(raw_values)[:, :, list(model.classes_).index(1)]
        else:
            values = np.asarray(raw_values)
    except Exception:
        importances = getattr(model, 'feature_importances_', np.ones(len(features)) / len(features))
        means = np.mean(sample, axis=0)
        stds = np.std(sample, axis=0) + 1e-9
        values = ((sample - means) / stds) * importances
    # Bin points by feature independently; do not export linkable patient vectors.
    points = []
    for i, feature in enumerate(features):
        span = np.ptp(sample[:, i])
        normalized = (sample[:, i] - sample[:, i].min()) / span if span else np.zeros(len(sample))
        counts, xedges, vedges = np.histogram2d(values[:, i], normalized, bins=(24, 5), range=[
            [float(values[:, i].min()) - 1e-9, float(values[:, i].max()) + 1e-9], [0, 1]])
        for a, b in zip(*np.nonzero(counts)):
            points.append(dict(Feature=feature, SHAP=float((xedges[a] + xedges[a + 1]) / 2),
                               Value=float((vedges[b] + vedges[b + 1]) / 2), Count=int(counts[a, b])))
    return dict(sample_size=len(sample), points=points,
                mean_absolute=dict(zip(features, np.abs(values).mean(axis=0).tolist())))
