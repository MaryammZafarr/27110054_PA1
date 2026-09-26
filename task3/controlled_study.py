
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task3.models.backbone import build_backbone, FEATURE_DIM
from task3.models.classifier_head import build_classifier
from task3.methods import dan_dg, sam
from shared.pacs_protocol import set_seed, save_json
from shared.pacs import PACS_CLASSES

N_CLASSES = len(PACS_CLASSES)


def run_controlled_study(source_loaders, val_loaders, steps_per_epoch, results_dir, cfg,
                          device="cuda", study="dan_dg"):
    sweep_results = {}
    common_kwargs = dict(max_epochs=cfg["max_epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          patience=cfg["patience"], device=device, seed=cfg["seed"])

    if study == "dan_dg":
        values = [0.1, 1.0, 10.0]
        for lam in values:
            set_seed(cfg["seed"])
            backbone = build_backbone(device=device)
            classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
            tag = f"dan_dg_lambda{lam}"
            result = dan_dg.train(backbone, classifier, source_loaders, val_loaders,
                                   steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                                   lambda_dg=lam, method_name=tag, **common_kwargs)
            sweep_results[str(lam)] = {"best_mean_val_macro_f1": result["best_mean_val_macro_f1"]}
    elif study == "sam":
        values = [0.01, 0.05, 0.1]
        for rho in values:
            set_seed(cfg["seed"])
            backbone = build_backbone(device=device)
            classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
            tag = f"sam_rho{rho}"
            result = sam.train(backbone, classifier, source_loaders, val_loaders,
                                steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                                rho=rho, method_name=tag, **common_kwargs)
            sweep_results[str(rho)] = {"best_mean_val_macro_f1": result["best_mean_val_macro_f1"]}
    else:
        raise ValueError(f"unknown study: {study}")

    save_json(sweep_results, os.path.join(results_dir, f"controlled_study_{study}_sweep.json"))
    _write_markdown_summary(sweep_results, study,
                             os.path.join(results_dir, f"controlled_study_{study}_sweep.md"))
    _plot_sweep(sweep_results, study, os.path.join(results_dir, "figures"))
    return sweep_results


def _write_markdown_summary(sweep_results, study, path):
    param_name = "lambda_DG" if study == "dan_dg" else "rho"
    lines = [f"| {param_name} | best_mean_val_macro_f1 |", "|---|---|"]
    for val, res in sweep_results.items():
        lines.append(f"| {val} | {res['best_mean_val_macro_f1']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _plot_sweep(sweep_results, study, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    xs = [float(k) for k in sweep_results.keys()]
    ys = [v["best_mean_val_macro_f1"] for v in sweep_results.values()]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    plt.xscale("log")
    plt.xlabel("lambda_DG" if study == "dan_dg" else "rho")
    plt.ylabel("best mean source val macro-F1")
    plt.title(f"Controlled study: {study}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"controlled_study_{study}_sweep.png"))
    plt.close()
