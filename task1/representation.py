
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def project_2d(features: np.ndarray, method: str = "tsne", seed: int = 6304, **kwargs):
    """
    Fit ONE 2D projection to the combined [clean; transformed] feature
    matrix so both conditions live in the same embedding space (do not fit
    separate projections and compare coordinates across them).
    features: (N_total, D) numpy array.
    """
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = kwargs.get("perplexity", 30)
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                        init="pca", learning_rate="auto")
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed,
                             n_neighbors=kwargs.get("n_neighbors", 15),
                             min_dist=kwargs.get("min_dist", 0.1))
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(features)


def plot_projection(coords_clean: np.ndarray, coords_transformed: np.ndarray,
                     labels_clean: np.ndarray, labels_transformed: np.ndarray,
                     class_names, title: str, save_path: str, method: str = "tsne"):
    """
    Color = ground-truth class, marker = clean ('o') vs transformed ('x').
    Both halves must come from ONE combined projection (see project_2d);
    this function only visualizes the already-fit coordinates.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")

    all_classes = np.unique(np.concatenate([labels_clean, labels_transformed]))
    for c in all_classes:
        color = cmap(int(c) % 10)
        m_clean = labels_clean == c
        m_trans = labels_transformed == c
        ax.scatter(coords_clean[m_clean, 0], coords_clean[m_clean, 1],
                   color=color, marker="o", s=18, alpha=0.7)
        ax.scatter(coords_transformed[m_trans, 0], coords_transformed[m_trans, 1],
                   color=color, marker="x", s=18, alpha=0.7)

    ax.set_title(f"{title} ({method})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    class_handles = [Line2D([0], [0], marker="s", color="w",
                             markerfacecolor=cmap(c % 10), markersize=8,
                             label=class_names[c]) for c in range(len(class_names))]
    marker_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", label="clean", markersize=8),
        Line2D([0], [0], marker="x", color="gray", linestyle="", label="transformed", markersize=8),
    ]
    leg1 = ax.legend(handles=class_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize=8, title="class")
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)