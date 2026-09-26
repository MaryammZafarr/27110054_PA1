# ATML Assignment 1

This repository contains the implementation and results for all four tasks of the ATML Assignment 1. The tasks study different aspects of model robustness, including visual cue reliance, domain adaptation, domain generalization, and open-set recognition.

## Repository Structure

```text
ATML_PA1/
│
├── README.md
│
├── task1/
│   ├── data/
│   ├── scripts/
│   ├── results/
│   └── ...
│
├── shared/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── training/
│   └── ...
│
├── task2/
│   ├── scripts/
│   ├── results/
│   └── ...
│
├── task3/
│   ├── scripts/
│   ├── results/
│   └── ...
│
└── task4/
    ├── data/
    ├── scripts/
    ├── results/
    └── ...
```

## Tasks

### Task 1 — Visual Cue Reliance

Task 1 studies how different models respond to changes in visual information while keeping the underlying object unchanged.

**Dataset:** STL-10

**Models:**

* ResNet-50
* ViT-B/16
* CLIP ViT-B/32

**Experiments:**

* Grayscale
* Hue rotation
* Translation
* Patch shuffling
* Shape–texture cue conflicts
* t-SNE / UMAP feature visualization

The goal is to examine whether models rely more on shape, color, or spatial information when making predictions.

---

### Tasks 2 & 3 — Domain Shift

Tasks 2 and 3 use a shared implementation because they use the same dataset, backbone, training setup, and evaluation procedure.

**Dataset:** PACS

**Source domains:** Photo, Art, Cartoon

**Target domain:** Sketch

**Backbone:** ResNet-18

**Shared setup includes:**

* Data loading and preprocessing
* Train/validation splitting
* ResNet-18 model
* Optimizer and training loop
* Checkpoint selection
* Accuracy and F1 evaluation
* Domain separability
* Confusion matrices
* Common utilities

#### Task 2 — Unsupervised Domain Adaptation

Task 2 compares source-only training with domain adaptation methods:

* ERM / Source-only
* DAN
* DANN
* CDAN

The target domain is available without labels during adaptation.

#### Task 3 — Domain Generalization

Task 3 studies whether models can generalize to the unseen Sketch domain without using Sketch during training.

Methods include:

* ERM
* DAN-DG
* SAM

DAN-DG aligns the source domains with each other, while SAM reduces sensitivity to small parameter perturbations.

---

### Task 4 — Open-Set Recognition

Task 4 studies how a classifier handles classes that were not present during training.

**Known classes:** CIFAR-10

**Unknown classes:** Selected CIFAR-100 classes

**Backbone:** ResNet-18

**Methods / Scores:**

* MSP
* MLS
* Energy
* Mahalanobis
* GCSC
* PROSER

The task evaluates both classification accuracy and the ability to reject unknown inputs.

**Evaluation includes:**

* Near vs. far unknowns
* AUROC
* Unknown rejection rates
* Bootstrap confidence intervals
* Failure-case analysis

---

## Shared Components

The `shared/` directory contains code used by Tasks 2 and 3, including:

```text
shared/
├── data/
│   └── PACS loading and preprocessing
│
├── models/
│   └── ResNet-18 and model utilities
│
├── training/
│   └── training and checkpoint utilities
│
└── utils/
    └── evaluation and analysis utilities
```

Task-specific code remains inside the corresponding `task2/` and `task3/` directories.

## Reproducibility

The experiments use fixed random seeds where required by the assignment.

**Seed:** `6304`

The repository contains the scripts used to train the models, run the experiments, and generate the reported results.

## Datasets

The experiments use:

* **STL-10** for Task 1
* **PACS** for Tasks 2 and 3
* **CIFAR-10 / CIFAR-100** for Task 4

Datasets are not included directly in the repository unless required by the assignment.

## Results

The detailed results, tables, figures, and analysis for each task are provided in the accompanying report.


| Task   | Problem               | Dataset      | Main Focus                     |
| ------ | --------------------- | ------------ | ------------------------------ |
| Task 1 | Visual cue reliance   | STL-10       | Shape, color, and spatial cues |
| Task 2 | Domain adaptation     | PACS         | Unlabeled target adaptation    |
| Task 3 | Domain generalization | PACS         | Unseen target domain           |
| Task 4 | Open-set recognition  | CIFAR-10/100 | Detecting unknown classes      |



