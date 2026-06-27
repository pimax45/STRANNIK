import numpy as np

from terrain_nav.scoring import (
    bias_corrected_rmse_score,
    dtw_similarity,
    fast_scores_batch,
    normalized_mutual_information,
    pearson_corr,
    score_profiles,
    slope_corr,
    spearman_corr,
)


def test_identical_and_offset_profiles_score_high():
    profile = np.array([2.0, 5.0, 9.0, 7.0, 12.0, 16.0])
    offset = profile + 25.0

    assert pearson_corr(profile, offset) > 0.999
    assert slope_corr(profile, offset) > 0.999
    assert spearman_corr(profile, offset) > 0.999
    assert bias_corrected_rmse_score(profile, offset) > 0.999
    assert normalized_mutual_information(profile, offset) > 0.9
    assert dtw_similarity(profile, offset) > 0.99
    score, _ = score_profiles(profile, offset, expensive=True)
    assert score > 0.95


def test_vectorized_score_prefers_matching_candidate():
    target = np.sin(np.linspace(0, 5, 80)) + 0.2 * np.linspace(0, 1, 80)
    candidates = np.vstack([target, target[::-1], np.zeros_like(target)])
    scores, valid = fast_scores_batch(target, candidates)

    assert np.argmax(scores) == 0
    assert scores[0] > 0.95
    np.testing.assert_allclose(valid, 1.0)

