import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from evaluation.common import K, pred_known
from evaluation.metrics import auroc
from evaluation.thresholds import calibrate_threshold
from utils import CIFAR10_CLASSES


def class_breakdown(b, u, tau):
    meta = b["meta"]
    pred = pred_known(b, "unk_all")
    uk = u["test"]
    rows = []
    for grp in ("near", "far"):
        for c in dict.fromkeys(meta["class_name"][meta["group"] == grp]):
            m = meta["class_name"] == c
            uc, pc = u["unk_all"][m], pred[m]
            acc = uc <= tau
            top_all = np.bincount(pc, minlength=K).argmax()
            top_acc = np.bincount(pc[acc], minlength=K).argmax() if acc.any() else -1
            rows.append(dict(group=grp, unknown_class=c, n=int(m.sum()), accept_rate=float(acc.mean()),
                             reject_rate=float(1 - acc.mean()), mean_u=float(uc.mean()),
                             auroc_vs_known=auroc(uk, uc),
                             top_pred_all=CIFAR10_CLASSES[top_all],
                             top_pred_all_share=float((pc == top_all).mean()),
                             top_pred_accepted=CIFAR10_CLASSES[top_acc] if top_acc >= 0 else "-",
                             top_pred_accepted_share=float((pc[acc] == top_acc).mean()) if acc.any() else 0.0))
    return pd.DataFrame(rows).sort_values(["group", "accept_rate"], ascending=[True, False]).reset_index(drop=True)


def absorption_matrix(b, u, tau):
    meta = b["meta"]
    pred = pred_known(b, "unk_all")
    acc = u["unk_all"] <= tau
    classes = list(dict.fromkeys(meta["class_name"]))
    M = np.zeros((len(classes), K), dtype=int)
    for i, c in enumerate(classes):
        m = (meta["class_name"] == c) & acc
        M[i] = np.bincount(pred[m], minlength=K)
    df = pd.DataFrame(M, index=classes, columns=CIFAR10_CLASSES)
    df.insert(0, "group", [meta["group"][meta["class_name"] == c][0] for c in classes])
    return df


def group_absorption(abs_df):
    out = {}
    for g in ("near", "far"):
        s = abs_df[abs_df.group == g][CIFAR10_CLASSES].sum()
        out[g] = (s / max(s.sum(), 1)).sort_values(ascending=False)
    return out


# ------------------------------------------------------------------------------------------------ RQ2
def score_agreement(b, U, tau, names=("MSP", "MLS", "Energy", "Mahalanobis")):
    res = {}
    pooled = {n: np.concatenate([U[n]["test"], U[n]["unk_all"]]) for n in names}
    unk = {n: U[n]["unk_all"] for n in names}
    near_mask = b["meta"]["group"] == "near"
    for label, sel in (("pooled_known+unknown", None), ("unknown_only", unk), ("near_only", {n: unk[n][near_mask] for n in names}),
                       ("far_only", {n: unk[n][~near_mask] for n in names})):
        src = pooled if sel is None else sel
        R = pd.DataFrame({a: [spearmanr(src[a], src[c])[0] for c in names] for a in names}, index=names)
        res[f"spearman_{label}"] = R
    # accept/reject agreement on unknowns at each score's own 95%-validation threshold
    acc = {n: unk[n] <= tau[n] for n in names}
    rows = []
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            rows.append(dict(A=a, B=c, both_accept=float((acc[a] & acc[c]).mean()), only_A_accepts=float((acc[a] & ~acc[c]).mean()),
                             only_B_accepts=float((~acc[a] & acc[c]).mean()), both_reject=float((~acc[a] & ~acc[c]).mean()),
                             B_rejects_given_A_accepts=float((~acc[c][acc[a]]).mean()) if acc[a].any() else np.nan,
                             A_rejects_given_B_accepts=float((~acc[a][acc[c]]).mean()) if acc[c].any() else np.nan))
    res["pairwise_accept_agreement_on_unknowns"] = pd.DataFrame(rows)
    # signal diagnostics: saturation of MSP, logit magnitude and feature norm
    fn = {s: np.linalg.norm(b[s]["feats"], axis=1) for s in ("val", "test", "near", "far")}
    ml = {s: b[s]["logits"][:, :K].max(1) for s in ("val", "test", "near", "far")}
    res["signal_diagnostics"] = pd.DataFrame({
        "msp_u_exactly_0": {s: float((U["MSP"][s] == 0).mean()) for s in fn},
        "mean_max_logit": {s: float(ml[s].mean()) for s in fn},
        "mean_feature_norm": {s: float(fn[s].mean()) for s in fn},
        "mean_u_Mahalanobis": {s: float(U["Mahalanobis"][s].mean()) for s in fn}})
    allf = np.concatenate([fn["test"], fn["near"], fn["far"]])
    allm = np.concatenate([ml["test"], ml["near"], ml["far"]])
    res["maxlogit_vs_featnorm_pearson"] = float(pearsonr(allf, allm)[0])
    # per-class rejection rate by score
    meta = b["meta"]
    rr = {}
    for n in names:
        rr[n] = {c: float((unk[n][meta["class_name"] == c] > tau[n]).mean()) for c in dict.fromkeys(meta["class_name"])}
    rr = pd.DataFrame(rr)
    rr.insert(0, "group", [meta["group"][meta["class_name"] == c][0] for c in rr.index])
    res["per_class_rejection_by_score"] = rr
    return res


# ------------------------------------------------------------------------------------------------ RQ3 / RQ4
def per_class_rejection(b, u, tau):
    meta = b["meta"]
    return pd.Series({c: float((u["unk_all"][meta["class_name"] == c] > tau).mean()) for c in dict.fromkeys(meta["class_name"])})


def compactness(b):

    f, y = b["train"]["feats"].astype(np.float64), b["train"]["labels"]
    mu = np.stack([f[y == c].mean(0) for c in range(K)])
    within = float(((f - mu[y]) ** 2).sum(1).mean())
    d = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    between = float(d[np.triu_indices(K, 1)].mean())
    z = np.sort(b["test"]["logits"][:, :K], 1)
    return dict(within_class_sqdist=within, between_class_sqdist=between, ratio=within / between,
                mean_top1_top2_margin_test=float((z[:, -1] - z[:, -2]).mean()),
                mean_feature_norm_test=float(np.linalg.norm(b["test"]["feats"], axis=1).mean()))


def proxy_detection(u, tau):
    """How well are VALIDATION manifold-mixup proxies (built from known data only) rejected at the calibrated
    threshold, versus the real near/far unknowns? (interpolation vs real-unknown gap)."""
    return dict(valmix_auroc=auroc(u["val"], u["valmix"]), valmix_reject=float((u["valmix"] > tau).mean()))


def dummy_wins_uncalibrated(b):
    if b["num_outputs"] <= K:
        return None
    return {s: float((b[s]["logits"][:, K:].max(1) > b[s]["logits"][:, :K].max(1)).mean())
            for s in ("val", "test", "near", "far", "valmix")}
