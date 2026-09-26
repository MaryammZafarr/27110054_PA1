
import os
import matplotlib.pyplot as plt


def plot_training_curves(history: list, method_name: str, save_dir: str):
    if not history:
        print(f"[plotting] WARNING: empty history for {method_name}, skipping curve plot.")
        return
    epochs = [h["epoch"] for h in history]
    cls_loss = [h["train_losses"].get("cls_loss", float("nan")) for h in history]
    mean_val_f1 = [h["mean_val_macro_f1"] for h in history]

    align_key = next((k for k in ("mmd_loss", "domain_loss") if k in history[0]["train_losses"]), None)
    align_vals = [h["train_losses"].get(align_key, float("nan")) for h in history] if align_key else None

    n_panels = 3 if align_key else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    axes[0].plot(epochs, cls_loss, marker="o")
    axes[0].set_title(f"{method_name}: classification loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cls_loss")

    axes[1].plot(epochs, mean_val_f1, marker="o", color="green")
    axes[1].set_title(f"{method_name}: mean source-val macro-F1")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro-F1")

    if align_key:
        axes[2].plot(epochs, align_vals, marker="o", color="orange")
        axes[2].set_title(f"{method_name}: {align_key}")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel(align_key)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{method_name}_curves.png"), dpi=150)
    plt.close(fig)


def plot_controlled_study(sweep_results: list, save_path: str):
    lambdas = [r["lambda_mmd"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, [r["mean_source_accuracy"] for r in sweep_results], marker="o", label="mean source acc")
    ax.plot(lambdas, [r["target_accuracy"] for r in sweep_results], marker="o", label="target acc")
    ax.plot(lambdas, [r["domain_separability_accuracy"] for r in sweep_results], marker="o", label="domain separability")
    ax.set_xscale("log")
    ax.set_xlabel("lambda_MMD"); ax.set_ylabel("score")
    ax.set_title("Controlled study: DAN lambda_MMD sweep")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)