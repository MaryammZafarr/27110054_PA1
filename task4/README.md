# Task 4 -- Open-Set Recognition (CIFAR-10 known / CIFAR-100 unknown)

Vanilla vs post-hoc scores (MSP, MLS, Energy, Mahalanobis) vs GCSC (RandAugment) vs PROSER (+ optional RPL).
Seed 6304 everywhere. Single command order: see `run_all.sh`.

---
## A. Things to go through BEFORE running

**A1. Reading (manual's list) -- what to focus on**
- Vaze et al. 2022: link between closed-set accuracy and Maximum Logit Score (why GCSC is a meaningful comparison).
- Zhou et al. 2021 (PROSER): Sec. 4.1 Eq. 5 (classifier placeholders, multiple dummies = max over dummies), Sec. 4.2 Eq. 6-7 + Alg. 2 (manifold mixup between different classes, mini-batch shuffle + mask), Sec. 4.3 (bias calibration to 95% validation acceptance), Suppl. II.2 (beta=1, gamma=0.1, C=5, alpha=2, lr 1e-3).
- Hendrycks & Gimpel 2017 (MSP), Liu et al. 2020 (Energy): only to justify the definitions of u_MSP / u_Energy.
- Chen et al. 2020 (RPL) only if you do the optional extension.
  **Re-read Sec. 3 of RPL before writing about it: the RPL code here follows the paper's ideas as I understand them and I could not re-read the PDF while writing it (see E2).**

**A2. Environment (Colab)**
1. Runtime -> GPU (T4 is enough). Mount Google Drive and `cd` into a Drive folder so `checkpoints/`, `cache/`, `results/` survive disconnects.
2. `pip install -r requirements.txt` (Colab already has most).
3. `python preflight.py --bench` -> confirms data/split/architecture/augmentation/config values vs the manual, proves training code never imports CIFAR-100, and prints projected time per 100 epochs. If `randaugment` is much slower than `base`, the CPU data loader is the bottleneck: raise `num_workers` in the configs.
4. `python sanity_check.py` (temporary; tiny end-to-end run in `sanity_tmp/`). Then **delete `sanity_check.py` and `sanity_tmp/`**.

**A3. Protocol rules to keep in your head**
- Unknowns (CIFAR-100) are touched ONLY by `extract_outputs.py` (after the freeze) and `evaluate_osr.py`. Never edit a config, checkpoint, score or threshold after seeing unknown results. If you must retrain, retrain everything affected and re-run extraction with `--refreeze` and say so in the report.
- Failure inspection is analysis only.
- Order: Vanilla -> (GCSC, PROSER) -> extract -> evaluate. PROSER needs `checkpoints/vanilla/best.pt`.
- Colab disconnect? Re-run the same command with `--resume` (restores model/optimizer/scheduler/scaler/history; the loader shuffle order restarts, which does not matter statistically).

**A4. Report bookkeeping (things reviewers look for)**
- State: single seed (so use the bootstrap CIs, not raw differences), fp16 AMP training + fp32 extraction, non-bit-wise determinism, drop_last, cosine per-iteration.
- State that CIFAR-100 was never used for training/checkpoint selection/score design/thresholds and cite `results/freeze_manifest.json`.
- Look at `results/fig_failures_vanilla_mls.png` yourself before calling a failure "plausible" or "surprising" and fill the `manual_note` column.

---
## B. Decisions I made (and why) -- copy into the report's "implementation details"

| # | Decision | Reason / effect |
|---|---|---|
| D1 | Split = `sklearn.train_test_split(stratify=y, random_state=6304, test_size=0.1)` -> 45,000 / 5,000 (500 per class), saved in `cache/splits/` and re-used by every method | identical data for all models; exactly 5,000 validation points for the 95th-percentile threshold |
| D2 | Validation: no augmentation. Test: full 10,000 CIFAR-10 test images (known). All normalisation uses CIFAR-10 train mean/std, also for unknowns | deployed model cannot re-normalise with unseen-data statistics |
| D3 | Model: torchvision ResNet-18 blocks, 3x3/s1 stem, no max-pool, kaiming init on the new conv (11,173,962 params); `forward_pre`/`forward_post` split at layer2/layer3 | manual spec; split needed for manifold mixup |
| D4 | Cosine decay stepped **per iteration** to 0, no warm-up, weight decay on all parameters, `drop_last=True`, fp16 AMP for training | manual leaves these open; drop_last keeps BN batches full and lets PROSER split 64/64. Extraction is always fp32 so logits are exact |
| D5 | Checkpoint = highest CIFAR-10 val accuracy, ties -> later epoch. PROSER: best among its 50 fine-tuning epochs, accuracy from the 10 known logits only | manual rule; PROSER CSA must exclude dummies |
| D6 | GCSC re-uses the Vanilla code; the only difference is `augmentation: randaugment` (`preflight.py` asserts the configs differ only there) | controlled comparison demanded by the manual |
| D7 | Mahalanobis: class means + ONE shared diagonal covariance = pooled within-class variance of unaugmented train-portion features, +1e-6 | standard shared-covariance estimator; total variance would include between-class spread |
| D8 | Scores in float64; Energy with T=1; MSP saturation (u exactly 0) is reported | avoids fake ties; documents when MSP loses resolution |
| D9 | PROSER loss = Eq.5 + gamma*Eq.7: `[z, max_c d_c]`; l1 = CE(.,y) + beta*CE(ground-truth logit masked to -1e9, dummy target); l2 = CE(mixed feature, dummy). One lambda~Beta(2,2) per batch (paper Alg. 2); partners = random other sample of the same half with a different label; unmatched rows dropped | verified against arXiv 2103.15086. Loss unit-checked against the closed form |
| D10 | PROSER init: known rows of `fc` copied from Vanilla, 5 dummy rows PyTorch-default random init (seeded); all layers fine-tuned, BN in train mode; batch halves = first 64 (classifier placeholders) / last 64 (data placeholders) | manual + paper |
| D11 | PROSER placeholder score u_PH = max dummy logit - max known logit; rejecting when u_PH > tau_95(val) is **exactly** the paper's "add a bias to the dummy logit so 95% of validation data are known" (bias = -tau), computed by exact percentile instead of the paper's interval search | manual: "calibrate using validation only". The MLS row of PROSER uses the first 10 logits only |
| D12 | Threshold: `np.percentile(u_val, 95)` (linear interpolation); accept iff u <= tau. FPR@95TPR = 1 - rejection | manual convention |
| D13 | AUROC: known = 10,000 test images vs near (800) / far (800) / all (1,600); unknown = positive class; rank-based, ties averaged. **Paired bootstrap (B=1000, seed 6304)** resamples known and unknown sets; the SAME resamples are used for all methods so differences are paired | 800 images per group + one seed -> raw differences of ~1 point are not interpretable without CIs |
| D14 | Failure selection (a priori): per group, 6 most-confident accepted unknowns (max 2 per class) + 6 random accepted unknowns; "plausible" flag comes from a semantic table fixed before running (`evaluation/failure_analysis.py::PLAUSIBLE`), with a manual-override column | avoids cherry-picking; you must eyeball the image grid |
| D15 | Extra analyses computed from the cache only: per-class breakdown, absorbing labels, score agreement (Spearman, rescue rates, MSP saturation, logit vs feature norm), feature geometry (within/between ratio, margin), validation manifold-mixup proxies, uncalibrated dummy-wins rate, per-class deltas | needed to answer RQ1-RQ4 with evidence, not speculation |
| D16 | Freeze protocol: `extract_outputs.py` hashes checkpoints into `results/freeze_manifest.json` BEFORE importing CIFAR-100; a changed checkpoint is refused; `train.py` asserts CIFAR-100 was never imported | provable "unknowns evaluated after everything is fixed" |
| D17 | RPL (optional): M=1 reciprocal point/class (512-d, N(0,0.1^2) init), logit = mean squared distance - dot product, CE + 0.1 * open-space term (d_e to own reciprocal point - learnable R)^2, u_RPL = -max logit. Same recipe as Vanilla (lr 0.1, 100 ep). `grad_clip` only if it diverges | see E2 |
| D18 | `cudnn.benchmark=True`, no forced determinism | speed; results reproducible statistically, not bit-wise |

---
## C. How to run
```bash
python preflight.py --bench
python sanity_check.py            # then delete it
bash run_all.sh                   # or the commands one by one (uncomment RPL lines if doing the extension)
```
Add `--resume` to any `train.py` command after a disconnect. Approximate cost on a T4 (use `--bench` for your machine): Vanilla and GCSC ~1 h each, PROSER ~0.5 h, RPL ~1 h.

## D. Evidence -> file -> research question
| Required evidence / RQ | File |
|---|---|
| Table: MSP/MLS/Energy/Mahalanobis on frozen Vanilla (AUROC near/far/all, tau, accept, reject, FPR) | `results/table1_posthoc_vanilla.{md,csv}` |
| Table: Vanilla/GCSC/PROSER(+placeholder row, +RPL) with CSA | `results/table2_trained_models.{md,csv}`, all 4 scores on all models: `table_supp_all_scores_all_models.csv` |
| Compact score-distribution + ROC figure (MSP, MLS, Mahalanobis) | `results/fig_scores_vanilla.png` (extra: `fig_scores_trained_models.png`) |
| >=3 near + >=3 far accepted failures with class, prediction, score, threshold | `results/failures_vanilla_mls.csv` + `fig_failures_vanilla_mls.png` (12 near + 12 far), `plausibility_vanilla_MLS.csv` |
| RQ1 semantic similarity | `class_breakdown_*.csv`, `absorption_vanilla_MLS.csv`, `fig_absorption_vanilla_MLS.png`, section RQ1 of `report_facts.md` |
| RQ2 what each score captures | `score_agreement_*.csv`, per-class rejection by score, section RQ2 of `report_facts.md` |
| RQ3 GCSC | Table 2 + `bootstrap_table2.csv` + `per_class_delta_reject_vs_vanilla.csv`, section RQ3 |
| RQ4 PROSER, CSA trade-off, interpolation insufficiency | Table 2, `geometry_and_proxies.csv`, `proser_uncalibrated_dummy_wins.json`, per-class deltas, section RQ4 |
| Protocol proof | `freeze_manifest.json`, `thresholds.json`, `evidence_checklist.md` |

## E. Limitations / things to verify
- E1. Single seed; bootstrap CIs capture test-set sampling noise, NOT training-seed variance. Say so.
- E2. RPL details (distance definition, open-space term, test score) are my reading of Chen et al.; PROSER was checked against the paper text. Verify RPL against the paper and adjust the description accordingly before citing it. If RPL diverges (NaN), set `grad_clip` and report the deviation.
- E3. Plausibility flags are a priori heuristics (e.g. mushroom->frog, wardrobe->truck are debatable): override in `manual_note`.
- E4. AUROC ranking and operating-point rejection can disagree; `report_facts.md` flags it automatically.
- E5. CIFAR-100 images are unaugmented 32x32 normalised with CIFAR-10 statistics.

## F. Layout
`configs/ methods/ (vanilla gcsc proser manifold_mixup rpl common) data/ (cifar10 cifar100_unknowns make_splits) models/ scores/ (msp mls energy mahalanobis placeholder) evaluation/ (metrics thresholds failure_analysis analysis plots common) train.py extract_outputs.py evaluate_osr.py preflight.py sanity_check.py(temporary) results/ cache/ checkpoints/`
