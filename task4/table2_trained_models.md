MLS is the common score; PROSER's placeholder score is an additional row. CSA uses only the ten known-class logits. tau from CIFAR-10 validation (95th pct).

| Method | CSA % | AUROC near % | AUROC far % | AUROC all % | tau | test accept % (known) | near reject % | far reject % | all reject % | FPR near % | FPR far % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla (MLS) | 94.8 | 80.95 [79.2, 82.5] | 89.56 [88.4, 90.7] | 85.25 [84.1, 86.2] | -5.737 | 94.7 | 31.4 | 54.5 | 42.9 | 68.6 | 45.5 |
| GCSC (MLS) | 95.2 | 82.34 [80.6, 83.8] | 90.55 [89.4, 91.6] | 86.44 [85.4, 87.4] | -6.101 | 94.9 | 36.0 | 58.9 | 47.4 | 64.0 | 41.1 |
| PROSER (MLS, known logits) | 94.2 | 80.73 [79.0, 82.2] | 88.80 [87.6, 89.9] | 84.77 [83.7, 85.8] | -4.309 | 94.9 | 32.8 | 49.9 | 41.3 | 67.2 | 50.1 |
| PROSER (placeholder score) | 94.2 | 79.02 [77.1, 80.7] | 87.97 [86.8, 89.2] | 83.50 [82.4, 84.6] | 0.771 | 95.2 | 29.9 | 42.0 | 35.9 | 70.1 | 58.0 |
