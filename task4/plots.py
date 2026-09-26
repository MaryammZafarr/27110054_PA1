import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from utils import CIFAR10_CLASSES

COL = {"known": "#2b7bba", "near": "#e08214", "far": "#c51b7d"}


def _tf(name, v):
    if name in ("MSP",):
        return np.log10(np.asarray(v) + 1e-8)      # u_MSP saturates at 0 -> spike at -8 shows the saturation
    if name in ("Mahalanobis",):
        return np.log10(np.asarray(v))
    return np.asarray(v)


def _xlabel(name):
    return {"MSP": "log10(1 - max softmax + 1e-8)", "Mahalanobis": "log10(min Mahalanobis dist.)",
            "MLS": "-max logit", "Energy": "-logsumexp(logits)", "Placeholder": "max dummy - max known logit"}[name]


def score_figure(panels, path, title):
    """panels: list of (panel_title, score_name, u_dict(test/near/far), tau). Row 1: histograms, row 2: ROC."""
    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 6.4))
    ax = np.atleast_2d(ax)
    for j, (ttl, name, u, tau) in enumerate(panels):
        allv = np.concatenate([_tf(name, u[s]) for s in ("test", "near", "far")])
        lo, hi = np.percentile(allv, 0.2), np.percentile(allv, 99.8)
        bins = np.linspace(lo, hi, 60)
        for s, lab in (("test", "known (C10 test)"), ("near", "near unk."), ("far", "far unk.")):
            ax[0, j].hist(np.clip(_tf(name, u[s]), lo, hi), bins=bins, density=True, alpha=0.5,
                          color=COL["known" if s == "test" else s], label=lab)
        ax[0, j].axvline(_tf(name, np.array([tau]))[0], color="k", ls="--", lw=1, label="tau (95% val)")
        ax[0, j].set_title(ttl, fontsize=10); ax[0, j].set_xlabel(_xlabel(name), fontsize=8)
        if j == 0:
            ax[0, j].set_ylabel("density"); ax[0, j].legend(fontsize=7)
        for s, lab in (("near", "known vs near"), ("far", "known vs far")):
            y = np.r_[np.zeros(len(u["test"])), np.ones(len(u[s]))]
            fpr, tpr, _ = roc_curve(y, np.r_[u["test"], u[s]])
            ax[1, j].plot(fpr, tpr, color=COL[s], label=lab)
        ax[1, j].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[1, j].set_xlabel("FPR"); ax[1, j].set_ylim(0, 1.02)
        if j == 0:
            ax[1, j].set_ylabel("TPR (unknown detected)"); ax[1, j].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def absorption_heatmap(abs_df, n_per_class, path, title):
    d = abs_df.drop(columns="group")
    frac = d.values / n_per_class
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(frac, cmap="magma_r", vmin=0, vmax=max(frac.max(), 1e-6))
    ax.set_xticks(range(len(d.columns))); ax.set_xticklabels(d.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(d.index)))
    ax.set_yticklabels([f"{c} ({g})" for c, g in zip(d.index, abs_df.group)])
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            if d.values[i, j]:
                ax.text(j, i, d.values[i, j], ha="center", va="center", fontsize=6,
                        color="w" if frac[i, j] > 0.5 * frac.max() else "k")
    ax.set_xlabel("predicted CIFAR-10 class (absorbs the accepted unknown)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="fraction of the class's images")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
