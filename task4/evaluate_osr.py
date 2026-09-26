
import argparse
import os

import numpy as np
import pandas as pd

from evaluation import analysis, failure_analysis, plots
from evaluation.common import compute_scores, csa, load_bundle
from evaluation.metrics import auroc_three, bootstrap_auroc, ci, rejection_stats
from evaluation.thresholds import calibrate_threshold
from utils import CIFAR10_CLASSES, SEED, ensure_dir, load_config, save_json

POSTHOC = ["MSP", "MLS", "Energy", "Mahalanobis"]
T2_ROWS = [("Vanilla (MLS)", "vanilla", "MLS"), ("GCSC (MLS)", "gcsc", "MLS"),
           ("PROSER (MLS, known logits)", "proser", "MLS"), ("PROSER (placeholder score)", "proser", "Placeholder"),
           ("RPL (max reciprocal-distance logit) [optional]", "rpl", "MLS")]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def verdict(d, lo, hi):
    if lo > 0:
        return "IMPROVEMENT (95% paired-bootstrap CI excludes 0)"
    if hi < 0:
        return "DEGRADATION (95% paired-bootstrap CI excludes 0)"
    return "no reliable difference (CI spans 0 -> within resampling noise; single seed)"


def build_table(rows, B_, U, bundles, tau, boot, ref_label):
    recs = []
    for label, model, score in rows:
        u, t = U[model][score], tau[model][score]
        au = auroc_three(u["test"], u["near"], u["far"])
        rs = rejection_stats(t, u["val"], u["test"], u["near"], u["far"])
        r = dict(method=label, CSA=csa(bundles[model]), AUROC_near=au["near"], AUROC_far=au["far"], AUROC_all=au["all"], **rs)
        for k in ("near", "far", "all"):
            lo, hi = ci(boot[label][k]); r[f"AUROC_{k}_lo"], r[f"AUROC_{k}_hi"] = lo, hi
            if ref_label is not None and label != ref_label:
                d = boot[label][k] - boot[ref_label][k]; dl, dh = ci(d)
                r[f"dAUROC_{k}_vs_ref"], r[f"dAUROC_{k}_lo"], r[f"dAUROC_{k}_hi"] = float(d.mean()), dl, dh
        recs.append(r)
    return pd.DataFrame(recs)


def pretty(df, with_csa):
    o = pd.DataFrame({"Method": df.method})
    if with_csa:
        o["CSA %"] = df.CSA.map(pct)
    for k in ("near", "far", "all"):
        o[f"AUROC {k} %"] = [f"{pct(a, 2)} [{pct(l, 1)}, {pct(h, 1)}]" for a, l, h in zip(df[f"AUROC_{k}"], df[f"AUROC_{k}_lo"], df[f"AUROC_{k}_hi"])]
    o["tau"] = df.tau.map(lambda v: f"{v:.3f}")
    o["test accept % (known)"] = df.known_test_accept.map(pct)
    o["near reject %"] = df.near_reject.map(pct)
    o["far reject %"] = df.far_reject.map(pct)
    o["all reject %"] = df.all_reject.map(pct)
    o["FPR near %"] = df.fpr_near.map(pct)
    o["FPR far %"] = df.fpr_far.map(pct)
    return o


def rank_note(df, col_auroc, col_rej, label):
    a = df.set_index("method")[col_auroc].rank(ascending=False)
    r = df.set_index("method")[col_rej].rank(ascending=False)
    bad = [m for m in a.index if a[m] != r[m]]
    return (f"{label}: AUROC-ranking and rejection-rate-ranking DISAGREE for {bad}" if bad
            else f"{label}: AUROC ranking and rejection-rate ranking agree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    R = ensure_dir(a.results_dir)
    cfg = load_config("configs/vanilla.yaml")
    have = [m for m in ("vanilla", "gcsc", "proser", "rpl") if os.path.exists(os.path.join(a.cache_dir, f"{m}_unknown.npz"))]
    assert "vanilla" in have, "run extract_outputs.py first"
    bundles = {m: load_bundle(a.cache_dir, m) for m in have}
    U = {m: compute_scores(bundles[m]) for m in have}

    # ------------------------------------------------------------- thresholds (validation only) -> frozen file
    tau = {m: {s: calibrate_threshold(U[m][s]["val"]) for s in U[m]} for m in have}
    save_json(tau, os.path.join(R, "thresholds.json"))

    # ------------------------------------------------------------- Table 1: post-hoc scores on frozen Vanilla
    ss1 = {s: (U["vanilla"][s]["test"], U["vanilla"][s]["near"], U["vanilla"][s]["far"]) for s in POSTHOC}
    boot1 = bootstrap_auroc(ss1, a.boot, SEED)
    rows1 = [(s, "vanilla", s) for s in POSTHOC]
    t1 = build_table([(l, m, s) for l, m, s in rows1], a.boot, U, bundles, tau, boot1, ref_label="MSP")
    t1.to_csv(os.path.join(R, "table1_posthoc_vanilla.csv"), index=False)
    p1 = pretty(t1, with_csa=False)
    with open(os.path.join(R, "table1_posthoc_vanilla.md"), "w") as f:
        f.write(f"Vanilla ResNet-18 (frozen). CIFAR-10 test accuracy (CSA) = {pct(csa(bundles['vanilla']), 2)}%. "
                f"AUROC with 95% bootstrap CI (B={a.boot}). tau = 95th percentile of u on CIFAR-10 validation.\n\n" + md_table(p1) + "\n")

    # ------------------------------------------------------------- Table 2: Vanilla vs GCSC vs PROSER (+RPL)
    rows2 = [r for r in T2_ROWS if r[1] in have]
    ss2 = {l: (U[m][s]["test"], U[m][s]["near"], U[m][s]["far"]) for l, m, s in rows2}
    boot2 = bootstrap_auroc(ss2, a.boot, SEED)
    t2 = build_table(rows2, a.boot, U, bundles, tau, boot2, ref_label="Vanilla (MLS)")
    t2.to_csv(os.path.join(R, "table2_trained_models.csv"), index=False)
    p2 = pretty(t2, with_csa=True)
    with open(os.path.join(R, "table2_trained_models.md"), "w") as f:
        f.write("MLS is the common score; PROSER's placeholder score is an additional row. CSA uses only the ten "
                "known-class logits. tau from CIFAR-10 validation (95th pct).\n\n" + md_table(p2) + "\n")

    # supplementary: all four post-hoc scores for every model
    sup = build_table([(f"{m}/{s}", m, s) for m in have for s in POSTHOC],
                       a.boot, U, bundles, tau,
                       bootstrap_auroc({f"{m}/{s}": (U[m][s]["test"], U[m][s]["near"], U[m][s]["far"]) for m in have for s in POSTHOC}, 200, SEED),
                       ref_label=None)
    sup.to_csv(os.path.join(R, "table_supp_all_scores_all_models.csv"), index=False)

    # bootstrap detail
    def boot_csv(boot, name):
        rec = []
        for lab, d in boot.items():
            for k, arr in d.items():
                lo, hi = ci(arr); rec.append(dict(method=lab, comparison=k, auroc_mean=arr.mean(), lo=lo, hi=hi))
        pd.DataFrame(rec).to_csv(os.path.join(R, name), index=False)
    boot_csv(boot1, "bootstrap_table1.csv"); boot_csv(boot2, "bootstrap_table2.csv")

    v = U["vanilla"]
    plots.score_figure([("MSP", "MSP", v["MSP"], tau["vanilla"]["MSP"]), ("MLS", "MLS", v["MLS"], tau["vanilla"]["MLS"]),
                        ("Mahalanobis", "Mahalanobis", v["Mahalanobis"], tau["vanilla"]["Mahalanobis"])],
                       os.path.join(R, "fig_scores_vanilla.png"), "Vanilla ResNet-18: score distributions (top) and ROC curves (bottom)")
    panels = [(l, s, U[m][s], tau[m][s]) for l, m, s in rows2]
    plots.score_figure([(p[0].split(" [")[0], p[1], p[2], p[3]) for p in panels], os.path.join(R, "fig_scores_trained_models.png"),
                       "Trained models: score distributions and ROC (MLS common score; PROSER placeholder)")

    bv = bundles["vanilla"]
    for s in POSTHOC:
        analysis.class_breakdown(bv, v[s], tau["vanilla"][s]).to_csv(os.path.join(R, f"class_breakdown_vanilla_{s}.csv"), index=False)
    for l, m, s in rows2:
        if m != "vanilla":
            analysis.class_breakdown(bundles[m], U[m][s], tau[m][s]).to_csv(os.path.join(R, f"class_breakdown_{m}_{s}.csv"), index=False)
    absv = analysis.absorption_matrix(bv, v["MLS"], tau["vanilla"]["MLS"])
    absv.to_csv(os.path.join(R, "absorption_vanilla_MLS.csv"))
    n_per = int((bv["meta"]["class_name"] == absv.index[0]).sum())
    plots.absorption_heatmap(absv, n_per, os.path.join(R, "fig_absorption_vanilla_MLS.png"),
                             "Vanilla + MLS: which CIFAR-10 label absorbs the ACCEPTED unknowns")
    gabs = analysis.group_absorption(absv)
    cbv = pd.read_csv(os.path.join(R, "class_breakdown_vanilla_MLS.csv"))

    from data.cifar100_unknowns import load_cifar100_test         
    x100, _, _ = load_cifar100_test(cfg["data_root"])
    fails = failure_analysis.select_failures(bv, v["MLS"]["unk_all"], tau["vanilla"]["MLS"])
    fails.drop(columns="_row").to_csv(os.path.join(R, "failures_vanilla_mls.csv"), index=False)
    failure_analysis.failure_figure(fails, {int(i): x100[int(i)] for i in fails.cifar100_index}, os.path.join(R, "fig_failures_vanilla_mls.png"),
                                    tau["vanilla"]["MLS"])
    plaus = failure_analysis.plausibility_summary(bv, v["MLS"]["unk_all"], tau["vanilla"]["MLS"])
    plaus.to_csv(os.path.join(R, "plausibility_vanilla_MLS.csv"), index=False)

    ag = analysis.score_agreement(bv, v, tau["vanilla"])
    for k, val in ag.items():
        if isinstance(val, pd.DataFrame):
            val.to_csv(os.path.join(R, f"score_agreement_{k}.csv"))
    save_json({"maxlogit_vs_featnorm_pearson": ag["maxlogit_vs_featnorm_pearson"]}, os.path.join(R, "score_agreement_scalars.json"))

    geo = {}
    for m in have:
        g = analysis.compactness(bundles[m])
        for l, mm, s in rows2:
            if mm == m:
                g_row = dict(g, **analysis.proxy_detection(U[m][s], tau[m][s]))
                geo[l] = g_row
    geo_df = pd.DataFrame(geo).T
    geo_df.to_csv(os.path.join(R, "geometry_and_proxies.csv"))
    dummy = analysis.dummy_wins_uncalibrated(bundles["proser"]) if "proser" in have else None
    if dummy:
        save_json(dummy, os.path.join(R, "proser_uncalibrated_dummy_wins.json"))
    pcv = analysis.per_class_rejection(bv, v["MLS"], tau["vanilla"]["MLS"])
    deltas = {}
    for l, m, s in rows2:
        if m != "vanilla":
            deltas[l] = analysis.per_class_rejection(bundles[m], U[m][s], tau[m][s]) - pcv
    if deltas:
        dd = pd.DataFrame(deltas)
        dd.insert(0, "vanilla_MLS_reject_rate", pcv)
        dd.insert(0, "group", [bv["meta"]["group"][bv["meta"]["class_name"] == c][0] for c in dd.index])
        dd.to_csv(os.path.join(R, "per_class_delta_reject_vs_vanilla.csv"))

    L = []
    L.append("# Task 4 -- numbers and interpretation guide (auto-generated; every claim below is computed from the cache)\n")
    vv = t2.set_index("method").loc["Vanilla (MLS)"]
    L.append("## Protocol audit")
    L.append(f"- models evaluated: {have}; seed {SEED}; thresholds = 95th percentile of u on CIFAR-10 validation (results/thresholds.json)")
    L.append(f"- unknowns per group: near={len(v['MLS']['near'])}, far={len(v['MLS']['far'])} (manual: 800 each); known test={len(v['MLS']['test'])}")
    L.append("- checkpoints hashed before unknowns were loaded: results/freeze_manifest.json\n")
    L.append("## What-to-watch: AUROC vs single operating point")
    L.append("- " + rank_note(t1, "AUROC_near", "near_reject", "Table 1 (near)"))
    L.append("- " + rank_note(t1, "AUROC_far", "far_reject", "Table 1 (far)"))
    L.append("- " + rank_note(t2, "AUROC_near", "near_reject", "Table 2 (near)"))
    L.append("- " + rank_note(t2, "AUROC_far", "far_reject", "Table 2 (far)"))
    L.append("  (if they disagree, say so in the report: AUROC ranks over ALL thresholds, rejection rate is one operating point at ~95% known acceptance)\n")

    L.append("## RQ1 -- semantic similarity (Vanilla + MLS, primary)")
    r_v = t1.set_index("method").loc["MLS"]
    L.append(f"- near AUROC {pct(r_v.AUROC_near, 2)} vs far AUROC {pct(r_v.AUROC_far, 2)}; near reject {pct(r_v.near_reject)}% vs far reject {pct(r_v.far_reject)}%")
    for g in ("near", "far"):
        sub = cbv[cbv.group == g]
        top = sub.head(3)
        L.append(f"- most often ACCEPTED {g} classes: " + ", ".join(f"{r.unknown_class} ({pct(r.accept_rate)}% accepted, mostly -> {r.top_pred_accepted})" for r in top.itertuples()))
        L.append(f"  least accepted {g}: " + ", ".join(f"{r.unknown_class} ({pct(r.accept_rate)}%)" for r in sub.tail(2).itertuples()))
        L.append(f"- CIFAR-10 labels absorbing accepted {g} unknowns (share): " + ", ".join(f"{k} {pct(x)}%" for k, x in gabs[g].head(4).items()))
    pl = plaus.groupby("group")[["accepted", "accepted_plausible", "accepted_surprising"]].sum()
    for g, r in pl.iterrows():
        L.append(f"- {g}: of {int(r.accepted)} accepted unknowns, {int(r.accepted_plausible)} are a-priori 'semantically plausible' confusions and "
                 f"{int(r.accepted_surprising)} are 'surprising' (see plausibility_vanilla_MLS.csv, failures_vanilla_mls.csv)")
    L.append("- guide: compare near vs far AUROC; name the absorbing labels (e.g. vehicles->truck/automobile, canids/felids->dog/cat) and contrast "
             "them with surprising far failures; check the image grid before labelling any failure 'plausible'.\n")

    L.append("## RQ2 -- what each score captures (Vanilla)")
    L.append("- AUROC near/far/all: " + "; ".join(f"{r.method}: {pct(r.AUROC_near)}/{pct(r.AUROC_far)}/{pct(r.AUROC_all)}" for r in t1.itertuples()))
    L.append(f"- MSP saturation (share of u_MSP exactly 0): " + ", ".join(f"{s}={pct(x, 2)}%" for s, x in ag["signal_diagnostics"]["msp_u_exactly_0"].items()))
    sp = ag["spearman_unknown_only"]
    L.append("- Spearman rank correlation among scores on unknowns only: " +
             ", ".join(f"{a}-{b}={sp.loc[a, b]:.2f}" for i, a in enumerate(POSTHOC) for b in POSTHOC[i + 1:]))
    L.append("- pairwise: share of unknowns accepted by A that B still rejects (rescue rate): " +
             "; ".join(f"{r.A} accepts -> {r.B} rejects {pct(r.B_rejects_given_A_accepts)}%, {r.B} accepts -> {r.A} rejects {pct(r.A_rejects_given_B_accepts)}%" for r in ag["pairwise_accept_agreement_on_unknowns"].itertuples()))
    sd = ag["signal_diagnostics"]
    L.append(f"- mean max-logit known/near/far = {sd.loc['test', 'mean_max_logit']:.2f}/{sd.loc['near', 'mean_max_logit']:.2f}/{sd.loc['far', 'mean_max_logit']:.2f}; "
             f"mean feature norm = {sd.loc['test', 'mean_feature_norm']:.2f}/{sd.loc['near', 'mean_feature_norm']:.2f}/{sd.loc['far', 'mean_feature_norm']:.2f}; "
             f"Pearson(max logit, feature norm) = {ag['maxlogit_vs_featnorm_pearson']:.2f}")
    prr = ag["per_class_rejection_by_score"]
    spread = (prr[POSTHOC].max(axis=1) - prr[POSTHOC].min(axis=1)).sort_values(ascending=False).head(3)
    L.append("- classes where the scores disagree most (max-min reject rate): " + ", ".join(f"{c} ({pct(x)} pts; best={prr.loc[c, POSTHOC].astype(float).idxmax()})" for c, x in spread.items()))
    L.append("- guide: MSP = normalised confidence (discards logit scale, saturates); MLS keeps absolute magnitude; Energy adds all logits "
             "(near-identical to MLS when one logit dominates); Mahalanobis uses feature geometry, independent of the classifier head.\n")

    if "gcsc" in have:
        L.append("## RQ3 -- GCSC (RandAugment) vs Vanilla, both with MLS")
        g = t2.set_index("method").loc["GCSC (MLS)"]
        L.append(f"- CSA: Vanilla {pct(vv.CSA, 2)}% -> GCSC {pct(g.CSA, 2)}% (delta {100 * (g.CSA - vv.CSA):+.2f} pts)")
        for k in ("near", "far", "all"):
            L.append(f"- AUROC {k}: {pct(vv[f'AUROC_{k}'], 2)} -> {pct(g[f'AUROC_{k}'], 2)}  paired delta {100 * g[f'dAUROC_{k}_vs_ref']:+.2f} pts "
                     f"[{100 * g[f'dAUROC_{k}_lo']:+.2f}, {100 * g[f'dAUROC_{k}_hi']:+.2f}] -> {verdict(0, g[f'dAUROC_{k}_lo'], g[f'dAUROC_{k}_hi'])}")
        L.append(f"- at the calibrated threshold: near reject {pct(vv.near_reject)}% -> {pct(g.near_reject)}%; far reject {pct(vv.far_reject)}% -> {pct(g.far_reject)}%")
        if "GCSC (MLS)" in deltas:
            dd = deltas["GCSC (MLS)"].sort_values()
            L.append("- per-class reject-rate change (GCSC - Vanilla): most worse: " + ", ".join(f"{c} {100 * x:+.1f}" for c, x in dd.head(3).items()) +
                     "; most better: " + ", ".join(f"{c} {100 * x:+.1f}" for c, x in dd.tail(3).items()))
        L.append("- guide: relate any CSA change to Vaze et al. (closed-set accuracy predicts MLS-OSR) but state whether the near/far AUROC actually moved; "
                 "explain why near unknowns (semantically overlapping with known classes) may benefit less than far ones; single seed -> use the CI verdict.\n")

    if "proser" in have:
        L.append("## RQ4 -- PROSER vs Vanilla and GCSC")
        for lab in ("PROSER (MLS, known logits)", "PROSER (placeholder score)"):
            r = t2.set_index("method").loc[lab]
            L.append(f"- {lab}: CSA {pct(r.CSA, 2)}% (delta vs Vanilla {100 * (r.CSA - vv.CSA):+.2f} pts); AUROC near/far/all = {pct(r.AUROC_near, 2)}/{pct(r.AUROC_far, 2)}/{pct(r.AUROC_all, 2)}")
            for k in ("near", "far", "all"):
                L.append(f"    paired delta AUROC {k} vs Vanilla-MLS: {100 * r[f'dAUROC_{k}_vs_ref']:+.2f} [{100 * r[f'dAUROC_{k}_lo']:+.2f}, {100 * r[f'dAUROC_{k}_hi']:+.2f}] -> {verdict(0, r[f'dAUROC_{k}_lo'], r[f'dAUROC_{k}_hi'])}")
            L.append(f"    near reject {pct(r.near_reject)}% / far reject {pct(r.far_reject)}% at known test acceptance {pct(r.known_test_accept)}%")
        L.append("- geometry / proxy evidence (geometry_and_proxies.csv): within/between-class ratio (smaller = tighter known regions), top1-top2 margin, "
                 "and how well VALIDATION manifold-mixup proxies are rejected:")
        for lab, r in geo_df.iterrows():
            L.append(f"    {lab}: ratio={r['ratio']:.3f}, margin={r['mean_top1_top2_margin_test']:.2f}, valmix AUROC={pct(r['valmix_auroc'])}, valmix reject={pct(r['valmix_reject'])}%")
        if dummy:
            L.append("- uncalibrated dummy-wins rate (bias=0), known val/test vs near/far/valmix: " + ", ".join(f"{k}={pct(x, 2)}%" for k, x in dummy.items()))
        if "PROSER (placeholder score)" in deltas:
            dd = deltas["PROSER (placeholder score)"].sort_values()
            L.append("- per-class reject change (PROSER placeholder - Vanilla MLS): most worse: " + ", ".join(f"{c} {100 * x:+.1f}" for c, x in dd.head(3).items()) +
                     "; most better: " + ", ".join(f"{c} {100 * x:+.1f}" for c, x in dd.tail(3).items()))
        L.append("- guide: if proxy (valmix) rejection is high but real near-unknown rejection barely moves, interpolation between known classes tightens "
                 "boundaries BETWEEN known classes but does not cover directions from which real unknowns arrive (e.g. a wolf lying inside dog's region).\n")
    if "rpl" in have:
        L.append("## Optional RPL: see the RPL row of Table 2 and RQ notes; compare its paired deltas with Vanilla/GCSC/PROSER.\n")
    with open(os.path.join(R, "report_facts.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ------------------------------------------------------------- evidence checklist
    need = [("Table 1 (post-hoc scores)", "table1_posthoc_vanilla.md"), ("Table 2 (Vanilla/GCSC/PROSER)", "table2_trained_models.md"),
            ("Figure: MSP/MLS/Mahalanobis", "fig_scores_vanilla.png"), ("Failures (>=3 near, >=3 far)", "failures_vanilla_mls.csv"),
            ("Failure image grid", "fig_failures_vanilla_mls.png"), ("RQ1 class breakdown", "class_breakdown_vanilla_MLS.csv"),
            ("RQ1 absorption", "absorption_vanilla_MLS.csv"), ("RQ2 agreement", "score_agreement_pairwise_accept_agreement_on_unknowns.csv"),
            ("RQ3/4 bootstrap", "bootstrap_table2.csv"), ("RQ4 geometry/proxies", "geometry_and_proxies.csv"),
            ("Freeze manifest", "freeze_manifest.json"), ("Report facts", "report_facts.md")]
    nn_ = int((fails.group == "near").sum()); nf_ = int((fails.group == "far").sum())
    lines = ["# Evidence checklist\n"] + [f"- [{'x' if os.path.exists(os.path.join(R, f)) else ' '}] {n}  ({f})" for n, f in need]
    lines.append(f"- [{'x' if nn_ >= 3 and nf_ >= 3 else ' '}] failure count near={nn_} far={nf_} (need >=3 each)")
    with open(os.path.join(R, "evidence_checklist.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(L[:12]))
    print("\nALL RESULTS WRITTEN TO", R)
    print(open(os.path.join(R, "table1_posthoc_vanilla.md")).read())
    print(open(os.path.join(R, "table2_trained_models.md")).read())


if __name__ == "__main__":
    main()
