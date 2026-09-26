Vanilla ResNet-18 (frozen). CIFAR-10 test accuracy (CSA) = 94.76%. AUROC with 95% bootstrap CI (B=1000). tau = 95th percentile of u on CIFAR-10 validation.

| Method | AUROC near % | AUROC far % | AUROC all % | tau | test accept % (known) | near reject % | far reject % | all reject % | FPR near % | FPR far % |
|---|---|---|---|---|---|---|---|---|---|---|
| MSP | 82.41 [80.9, 83.9] | 89.76 [88.7, 90.7] | 86.08 [85.1, 87.0] | 0.166 | 94.6 | 28.4 | 45.2 | 36.8 | 71.6 | 54.8 |
| MLS | 80.95 [79.2, 82.5] | 89.56 [88.4, 90.7] | 85.25 [84.1, 86.2] | -5.737 | 94.7 | 31.4 | 54.5 | 42.9 | 68.6 | 45.5 |
| Energy | 80.96 [79.2, 82.5] | 89.60 [88.4, 90.7] | 85.28 [84.1, 86.3] | -5.921 | 94.8 | 31.1 | 54.5 | 42.8 | 68.9 | 45.5 |
| Mahalanobis | 79.43 [77.8, 81.0] | 91.44 [90.7, 92.3] | 85.44 [84.5, 86.4] | 2723.517 | 94.6 | 28.1 | 54.2 | 41.2 | 71.9 | 45.8 |
