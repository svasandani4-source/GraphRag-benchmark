from grb.eval.metrics import all_support, bootstrap_ci, exact_match, f1_score, support_recall


def test_em_normalises_articles_and_punctuation():
    assert exact_match("The Dun Mountains.", "dun mountains") == 1.0


def test_f1_partial_and_yes_no():
    assert 0 < f1_score("Mount Arrow peak", "Mount Arrow") < 1
    assert f1_score("yes", "no") == 0.0


def test_support_metrics():
    assert support_recall(["a", "b"], ["a", "c"]) == 0.5
    assert all_support(["a", "c", "d"], ["a", "c"]) == 1.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi
