
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.common import pred_known
from utils import CIFAR10_CLASSES, SEED

PLAUSIBLE = {  # unknown CIFAR-100 class -> semantically plausible CIFAR-10 predictions (fixed a priori)
    "bus": {"truck", "automobile"}, "pickup_truck": {"truck", "automobile"}, "motorcycle": {"automobile", "truck"},
    "tractor": {"truck", "automobile"}, "wolf": {"dog", "cat", "deer"}, "fox": {"dog", "cat"},
    "leopard": {"cat", "dog"}, "camel": {"horse", "deer"},
    "bottle": set(), "bowl": set(), "chair": set(), "clock": set(), "keyboard": set(),
    "mushroom": {"frog"}, "sunflower": set(), "wardrobe": {"truck"},
}


def select_failures(b, u_unk, tau, n_conf=6, n_rand=6, per_class_cap=2, seed=SEED):
    meta = b["meta"]
    pred = pred_known(b, "unk_all")
    logits = b["unk_all"]["logits"][:, :10]
    rng = np.random.RandomState(seed)
    rows = []
    for grp in ("near", "far"):
        cand = np.where((meta["group"] == grp) & (u_unk <= tau))[0]
        cand = cand[np.argsort(u_unk[cand])]
        chosen, cnt = [], {}
        for i in cand:
            c = meta["class_name"][i]
            if cnt.get(c, 0) < per_class_cap:
                chosen.append(i); cnt[c] = cnt.get(c, 0) + 1
            if len(chosen) == n_conf:
                break
        rest = np.array([i for i in cand if i not in set(chosen)])
        rnd = list(rng.choice(rest, min(n_rand, len(rest)), replace=False)) if len(rest) else []
        for kind, ids in (("most_confident", chosen), ("random", rnd)):
            for i in ids:
                top2 = np.argsort(logits[i])[::-1][:2]
                cn = meta["class_name"][i]
                rows.append(dict(group=grp, selection=kind, unknown_class=cn, cifar100_index=int(meta["cifar100_index"][i]),
                                 predicted_class=CIFAR10_CLASSES[pred[i]], second_class=CIFAR10_CLASSES[top2[1]],
                                 score_u=float(u_unk[i]), threshold_tau=float(tau), margin_tau_minus_u=float(tau - u_unk[i]),
                                 max_logit=float(logits[i].max()),
                                 auto_flag="plausible" if CIFAR10_CLASSES[pred[i]] in PLAUSIBLE[cn] else "surprising",
                                 manual_note="", _row=int(i)))
    return pd.DataFrame(rows)


def plausibility_summary(b, u_unk, tau):
    meta = b["meta"]
    pred = pred_known(b, "unk_all")
    rows = []
    for c in dict.fromkeys(meta["class_name"]):
        m = (meta["class_name"] == c) & (u_unk <= tau)
        pl = sum(CIFAR10_CLASSES[p] in PLAUSIBLE[c] for p in pred[m])
        rows.append(dict(group=meta["group"][meta["class_name"] == c][0], unknown_class=c, accepted=int(m.sum()),
                         accepted_plausible=int(pl), accepted_surprising=int(m.sum() - pl)))
    return pd.DataFrame(rows)


def failure_figure(df, images, path, tau):
    ncol = 6
    groups = [("near", "most_confident"), ("near", "random"), ("far", "most_confident"), ("far", "random")]
    fig, axes = plt.subplots(len(groups), ncol, figsize=(2.1 * ncol, 2.5 * len(groups)))
    for r, (g, k) in enumerate(groups):
        sub = df[(df.group == g) & (df.selection == k)].reset_index(drop=True)
        for c in range(ncol):
            ax = axes[r, c]; ax.axis("off")
            if c < len(sub):
                row = sub.iloc[c]
                ax.imshow(images[row.cifar100_index])
                ax.set_title(f"{row.unknown_class} -> {row.predicted_class}\nu={row.score_u:.2f} (tau={tau:.2f})\n[{row.auto_flag}]",
                             fontsize=7, color="#1a7f37" if row.auto_flag == "plausible" else "#b42318")
        axes[r, 0].text(-0.05, 0.5, f"{g}\n{k}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Incorrectly ACCEPTED unknowns under the Vanilla-MLS threshold", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
