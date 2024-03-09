import numpy as np
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc

from itertools import accumulate
from functools import partial


def EER(labels, scores):
    """
    labels: (N,1) value: 0,1
    scores: (N,1) value: -1 ~ 1
    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    s = interp1d(fpr, tpr)
    a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    eer = brentq(a, 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh


def eer_yist_f(labels, scores):
    """
    Args:
        labels: (N,1) with value being 0 or 1
        scores: (N,1) within [-1, 1]
    Returns:
        equal_error_rates
        threshold
    """
    joints = sorted(zip(scores, labels), key=lambda x: x[0])

    sorted_scores, sorted_labels = zip(*joints)

    total_ones = sum(sorted_labels)
    total_zeros = len(sorted_labels) - total_ones

    prefsum_ones = list(
        accumulate(sorted_labels, partial(_count_labels, label_to_count=1), initial=0)
    )
    prefsum_zeros = list(
        accumulate(sorted_labels, partial(_count_labels, label_to_count=0), initial=0)
    )

    ext_scores = [-1.0, *sorted_scores, 1.0]

    thresh_left, thresh_right = 0, len(ext_scores)

    while True:
        if thresh_left == thresh_right:
            break

        thresh_idx = (thresh_left + thresh_right) // 2
        nb_false_positives = total_zeros - prefsum_zeros[thresh_idx]
        nb_false_negatives = prefsum_ones[thresh_idx]

        if nb_false_positives > nb_false_negatives:
            thresh_left = thresh_idx
        elif nb_false_positives < nb_false_negatives:
            thresh_right = thresh_idx
        else:
            break

    thresh = (ext_scores[thresh_idx] + ext_scores[thresh_idx + 1]) / 2
    false_negative_ratio = nb_false_negatives / len(labels)
    false_positive_ratio = nb_false_positives / len(labels)
    equal_error_rate = (false_positive_ratio + false_negative_ratio) / 2

    return equal_error_rate, thresh


def _count_labels(counted_so_far, label, label_to_count=0):
    return counted_so_far + 1 if label == label_to_count else counted_so_far
