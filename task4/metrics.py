import numpy as np
from scipy.stats import rankdata


def auroc(u_known, u_unknown):
    """P(u_unknown > u_known) + 0.5 P(tie). Positive class = unknown, larger u = more novel."""
    n0, n1 = len(u_known), len(u_unknown)
    r = rankdata(np.concatenate([u_known, u_unknown]))
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def auroc_three(u_known, u_near, u_far):
    return dict(near=auroc(u_known, u_near), far=auroc(u_known, u_far),
                all=auroc(u_known, np.concatenate([u_near, u_far])))


def rejection_stats(tau, u_val, u_test, u_near, u_far):
    """Accept iff u <= tau. Returns acceptance of known val/test and rejection of near/far/all unknowns.
    FPR@95TPR convention of the manual = fraction of unknowns wrongly ACCEPTED at this threshold."""
    acc_val, acc_test = float(np.mean(u_val <= tau)), float(np.mean(u_test <= tau))
    rej_near, rej_far = float(np.mean(u_near > tau)), float(np.mean(u_far > tau))
    rej_all = float(np.mean(np.concatenate([u_near, u_far]) > tau))
    return dict(tau=float(tau), known_val_accept=acc_val, known_test_accept=acc_test,
                near_reject=rej_near, far_reject=rej_far, all_reject=rej_all,
                fpr_near=1 - rej_near, fpr_far=1 - rej_far, fpr_all=1 - rej_all)


def bootstrap_auroc(score_sets, B=1000, seed=6304):
    """Paired percentile bootstrap. score_sets: {name: (u_known_test, u_near, u_far)}.

    The SAME resampled indices are used for every method in a replicate, so differences between methods are
    paired (much tighter than comparing two independent CIs). 'all' = resampled near U resampled far.
    Returns {name: {near/far/all: array(B)}}.
    """
    rng = np.random.RandomState(seed)
    names = list(score_sets)
    nk, nn, nf = (len(score_sets[names[0]][i]) for i in range(3))
    out = {n: {"near": np.empty(B), "far": np.empty(B), "all": np.empty(B)} for n in names}
    for b in range(B):
        ik, i_n, i_f = rng.randint(0, nk, nk), rng.randint(0, nn, nn), rng.randint(0, nf, nf)
        for n in names:
            uk, un, uf = score_sets[n]
            k, a, f = uk[ik], un[i_n], uf[i_f]
            out[n]["near"][b] = auroc(k, a)
            out[n]["far"][b] = auroc(k, f)
            out[n]["all"][b] = auroc(k, np.concatenate([a, f]))
    return out


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))
