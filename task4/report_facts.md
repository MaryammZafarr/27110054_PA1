# Task 4 -- numbers and interpretation guide (auto-generated; every claim below is computed from the cache)

## Protocol audit
- models evaluated: ['vanilla', 'gcsc', 'proser']; seed 6304; thresholds = 95th percentile of u on CIFAR-10 validation (results/thresholds.json)
- unknowns per group: near=800, far=800 (manual: 800 each); known test=10000
- checkpoints hashed before unknowns were loaded: results/freeze_manifest.json

## What-to-watch: AUROC vs single operating point
- Table 1 (near): AUROC-ranking and rejection-rate-ranking DISAGREE for ['MSP', 'MLS']
- Table 1 (far): AUROC-ranking and rejection-rate-ranking DISAGREE for ['MSP', 'MLS', 'Energy', 'Mahalanobis']
- Table 2 (near): AUROC-ranking and rejection-rate-ranking DISAGREE for ['Vanilla (MLS)', 'PROSER (MLS, known logits)']
- Table 2 (far): AUROC ranking and rejection-rate ranking agree
  (if they disagree, say so in the report: AUROC ranks over ALL thresholds, rejection rate is one operating point at ~95% known acceptance)

## RQ1 -- semantic similarity (Vanilla + MLS, primary)
- near AUROC 80.95 vs far AUROC 89.56; near reject 31.4% vs far reject 54.5%
- most often ACCEPTED near classes: pickup_truck (91.0% accepted, mostly -> automobile), bus (83.0% accepted, mostly -> truck), wolf (77.0% accepted, mostly -> cat)
  least accepted near: tractor (52.0%), motorcycle (41.0%)
- CIFAR-10 labels absorbing accepted near unknowns (share): truck 27.5%, cat 18.4%, automobile 17.9%, dog 12.2%
- most often ACCEPTED far classes: wardrobe (70.0% accepted, mostly -> truck), mushroom (53.0% accepted, mostly -> bird), chair (46.0% accepted, mostly -> airplane)
  least accepted far: clock (35.0%), keyboard (32.0%)
- CIFAR-10 labels absorbing accepted far unknowns (share): cat 28.3%, truck 18.1%, bird 17.9%, airplane 12.1%
- far: of 364 accepted unknowns, 47 are a-priori 'semantically plausible' confusions and 317 are 'surprising' (see plausibility_vanilla_MLS.csv, failures_vanilla_mls.csv)
- near: of 549 accepted unknowns, 420 are a-priori 'semantically plausible' confusions and 129 are 'surprising' (see plausibility_vanilla_MLS.csv, failures_vanilla_mls.csv)
- guide: compare near vs far AUROC; name the absorbing labels (e.g. vehicles->truck/automobile, canids/felids->dog/cat) and contrast them with surprising far failures; check the image grid before labelling any failure 'plausible'.

## RQ2 -- what each score captures (Vanilla)
- AUROC near/far/all: MSP: 82.4/89.8/86.1; MLS: 81.0/89.6/85.3; Energy: 81.0/89.6/85.3; Mahalanobis: 79.4/91.4/85.4
- MSP saturation (share of u_MSP exactly 0): val=0.00%, test=0.00%, near=0.00%, far=0.00%
- Spearman rank correlation among scores on unknowns only: MSP-MLS=0.92, MSP-Energy=0.89, MSP-Mahalanobis=0.93, MLS-Energy=1.00, MLS-Mahalanobis=0.83, Energy-Mahalanobis=0.81
- pairwise: share of unknowns accepted by A that B still rejects (rescue rate): MSP accepts -> MLS rejects 17.0%, MLS accepts -> MSP rejects 8.1%; MSP accepts -> Energy rejects 18.5%, Energy accepts -> MSP rejects 9.9%; MSP accepts -> Mahalanobis rejects 11.6%, Mahalanobis accepts -> MSP rejects 5.0%; MLS accepts -> Energy rejects 1.6%, Energy accepts -> MLS rejects 1.9%; MLS accepts -> Mahalanobis rejects 12.9%, Mahalanobis accepts -> MLS rejects 15.5%; Energy accepts -> Mahalanobis rejects 14.1%, Mahalanobis accepts -> Energy rejects 16.5%
- mean max-logit known/near/far = 9.18/6.85/5.79; mean feature norm = 6.05/5.54/5.58; Pearson(max logit, feature norm) = 0.79
- classes where the scores disagree most (max-min reject rate): wardrobe (28.0 pts; best=Mahalanobis), chair (20.0 pts; best=Energy), keyboard (18.0 pts; best=MLS)
- guide: MSP = normalised confidence (discards logit scale, saturates); MLS keeps absolute magnitude; Energy adds all logits (near-identical to MLS when one logit dominates); Mahalanobis uses feature geometry, independent of the classifier head.

## RQ3 -- GCSC (RandAugment) vs Vanilla, both with MLS
- CSA: Vanilla 94.76% -> GCSC 95.21% (delta +0.45 pts)
- AUROC near: 80.95 -> 82.34  paired delta +1.37 pts [+0.01, +2.73] -> IMPROVEMENT (95% paired-bootstrap CI excludes 0)
- AUROC far: 89.56 -> 90.55  paired delta +1.00 pts [-0.10, +2.18] -> no reliable difference (CI spans 0 -> within resampling noise; single seed)
- AUROC all: 85.25 -> 86.44  paired delta +1.19 pts [+0.27, +2.07] -> IMPROVEMENT (95% paired-bootstrap CI excludes 0)
- at the calibrated threshold: near reject 31.4% -> 36.0%; far reject 54.5% -> 58.9%
- per-class reject-rate change (GCSC - Vanilla): most worse: clock -4.0, keyboard -2.0, tractor -2.0; most better: chair +11.0, sunflower +11.0, bottle +15.0
- guide: relate any CSA change to Vaze et al. (closed-set accuracy predicts MLS-OSR) but state whether the near/far AUROC actually moved; explain why near unknowns (semantically overlapping with known classes) may benefit less than far ones; single seed -> use the CI verdict.

## RQ4 -- PROSER vs Vanilla and GCSC
- PROSER (MLS, known logits): CSA 94.24% (delta vs Vanilla -0.52 pts); AUROC near/far/all = 80.73/88.80/84.77
    paired delta AUROC near vs Vanilla-MLS: -0.23 [-1.02, +0.58] -> no reliable difference (CI spans 0 -> within resampling noise; single seed)
    paired delta AUROC far vs Vanilla-MLS: -0.75 [-1.36, -0.10] -> DEGRADATION (95% paired-bootstrap CI excludes 0)
    paired delta AUROC all vs Vanilla-MLS: -0.49 [-1.01, +0.01] -> no reliable difference (CI spans 0 -> within resampling noise; single seed)
    near reject 32.8% / far reject 49.9% at known test acceptance 94.9%
- PROSER (placeholder score): CSA 94.24% (delta vs Vanilla -0.52 pts); AUROC near/far/all = 79.02/87.97/83.50
    paired delta AUROC near vs Vanilla-MLS: -1.93 [-2.87, -1.05] -> DEGRADATION (95% paired-bootstrap CI excludes 0)
    paired delta AUROC far vs Vanilla-MLS: -1.56 [-2.31, -0.84] -> DEGRADATION (95% paired-bootstrap CI excludes 0)
    paired delta AUROC all vs Vanilla-MLS: -1.75 [-2.36, -1.15] -> DEGRADATION (95% paired-bootstrap CI excludes 0)
    near reject 29.9% / far reject 42.0% at known test acceptance 95.2%
- geometry / proxy evidence (geometry_and_proxies.csv): within/between-class ratio (smaller = tighter known regions), top1-top2 margin, and how well VALIDATION manifold-mixup proxies are rejected:
    Vanilla (MLS): ratio=0.046, margin=8.03, valmix AUROC=60.0, valmix reject=15.4%
    GCSC (MLS): ratio=0.081, margin=7.87, valmix AUROC=67.9, valmix reject=20.1%
    PROSER (MLS, known logits): ratio=0.050, margin=7.59, valmix AUROC=53.6, valmix reject=11.3%
    PROSER (placeholder score): ratio=0.050, margin=7.59, valmix AUROC=53.7, valmix reject=10.2%
- uncalibrated dummy-wins rate (bias=0), known val/test vs near/far/valmix: val=8.98%, test=9.19%, near=45.62%, far=61.75%, valmix=18.14%
- per-class reject change (PROSER placeholder - Vanilla MLS): most worse: bottle -22.0, clock -19.0, bowl -17.0; most better: leopard +5.0, pickup_truck +7.0, camel +9.0
- guide: if proxy (valmix) rejection is high but real near-unknown rejection barely moves, interpolation between known classes tightens boundaries BETWEEN known classes but does not cover directions from which real unknowns arrive (e.g. a wolf lying inside dog's region).

