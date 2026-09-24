# CANN

Alignment-free viral family classification using persistent commutative-algebraic descriptors and neural representation learning.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
- [Code Overview](#code-overview)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Outputs](#outputs)
- [Acknowledgments](#acknowledgments)

## Introduction

CANN combines persistent facet-number descriptors for nucleotide k-mers of lengths 3, 4, and 5 with relative-frequency features for k-mers of lengths 6 and 7.

CNN and Transformer encoders are trained independently. Cosine distances from the two embedding spaces are averaged with equal weights for nearest-neighbor classification.

Consensus 1-NN is the principal prediction method. Results for 5-NN and the individual encoders are also evaluated.

The main benchmark protocol uses stratified five-fold cross-validation repeated over 30 random seeds. Held-out predictions are pooled across the five folds within each seed, and performance is summarized across the 30 seeds.

## Prerequisites

The recorded software environment is:

| Software | Version |
| --- | --- |
| Python | 3.11.3 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| pandas | 2.3.0 |
| scikit-learn | 1.7.0 |
| PyTorch | 2.1.2+cpu |
| Biopython | 1.86 |
| GUDHI | 3.11.0 |

Operating system:

```text
Linux-5.15.0-187-generic-x86_64-with-glibc2.35
```

The recorded PyTorch build uses the CPU. CUDA and a GPU are not required for the instructions below.

## Installation

Download the repository and open a terminal in its directory.

Using Python 3.11.3, create and activate an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

Install the CPU version of PyTorch:

```bash
python -m pip install torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Run all subsequent commands from the repository directory with this environment activated.

## Datasets

The main benchmarks use NCBI 2020, NCBI 2022, NCBI 2024, and NCBI 2024 All.

Dataset download:

**[Add a stable link to the exact FASTA files and family-label metadata.]**

Each dataset requires:

1. A nucleotide FASTA file.
2. A metadata CSV containing these columns:

| Column | Description |
| --- | --- |
| `Accession (version)` | Genome accession including its version |
| `Family` | Viral family label |

Accessions must match between the FASTA and metadata.

Use the same sequence snapshots and family assignments as the corresponding manuscript benchmark. Updated database downloads may contain different sequences or labels.

The supplied preprocessing handles sequence cleaning, accession matching, and duplicate sequences. The main benchmark configuration retains families with at least 15 genomes.

This workflow covers the main cross-validation experiments. Separate 2026 reference-based evaluations and feature-ablation experiments require their corresponding scripts.

## Code Overview

| File | Purpose |
| --- | --- |
| `cann.py` | Count usable genomes, generate features, and validate feature files |
| `psrt.py` | Persistent descriptor calculations |
| `kmer_only_encoder.py` | Nucleotide k-mer counting |
| `config.py` | Data paths, model settings, and evaluation configuration |
| `data.py` | Data cleaning, feature loading, and normalization |
| `dataset.py` | Datasets, balanced sampling, and data loaders |
| `cnn.py` | CNN encoder |
| `transformer.py` | Transformer encoder |
| `training.py` | Encoder training and validation |
| `evaluation.py` | Individual-encoder and consensus evaluation |
| `5CV.py` | Repeated stratified five-fold cross-validation |
| `results.py` | Result aggregation |

Keep these Python files together in the repository directory.

## Reproducing the Experiments

### 1. Configure the dataset paths

Create a local `datasets` directory:

```bash
mkdir -p datasets
```

For the example commands below, place the selected dataset files there as:

- `datasets/sequences.fasta`
- `datasets/metadata.csv`

Set the path section of `config.py` as follows. Preserve the model, feature, and cross-validation settings below that section.

```python
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

BASE_DIR = PROJECT_DIR / "datasets"
CSV_PATH = BASE_DIR / "metadata.csv"
FASTA_PATH = BASE_DIR / "sequences.fasta"

PSRT_BASE = PROJECT_DIR / "features"
KMER_COUNT_ROOT = PSRT_BASE / "output_kmer" / "kmer_counts"

RESULT_DIR = PROJECT_DIR / "outputs"
```

Defining `FASTA_PATH` in `config.py` overrides the fallback path in `data.py`.

Use separate feature and result directories for each benchmark dataset.

### 2. Count usable genomes

```bash
python cann.py \
    --mode count \
    --fasta datasets/sequences.fasta \
    --out_root features
```

The output includes:

```text
USABLE_GENOMES=N
```

Here, `N` is the number of cleaned unique genomes available for feature generation.

### 3. Generate features

Each worker processes one cleaned unique genome. Task indices start at zero.

For example, process the first genome with:

```bash
python cann.py \
    --mode worker \
    --fasta datasets/sequences.fasta \
    --out_root features \
    --task_id 0
```

To process every genome sequentially, run the following in Bash:

```bash
set -euo pipefail

count_output=$(python cann.py \
    --mode count \
    --fasta datasets/sequences.fasta \
    --out_root features)

genome_count=$(printf '%s\n' "$count_output" \
    | awk -F= '/^USABLE_GENOMES=/{print $2}')

if [[ ! "$genome_count" =~ ^[0-9]+$ ]] || (( genome_count == 0 )); then
    echo "A positive usable-genome count is required." >&2
    exit 1
fi

for ((task_id = 0; task_id < genome_count; task_id++)); do
    python cann.py \
        --mode worker \
        --fasta datasets/sequences.fasta \
        --out_root features \
        --task_id "$task_id"
done
```

Full-dataset feature generation may take substantial time. Workers can also be scheduled independently on a computing cluster.

The five feature blocks contain:

| Feature block | Scalar values per genome |
| --- | ---: |
| Facet descriptors, k=3 | 128 |
| Facet descriptors, k=4 | 512 |
| Facet descriptors, k=5 | 2,048 |
| k-mer composition, k=6 | 4,096 |
| k-mer composition, k=7 | 16,384 |
| **Total** | **23,168** |

Facet files are stored under:

```text
features/output_psrt/all_features/k3/
features/output_psrt/all_features/k4/
features/output_psrt/all_features/k5/
```

Each file is named `<accession>_facet0.npy` and has shape `(4**k, 2)`.

K-mer count files are stored under:

```text
features/output_kmer/kmer_counts/6/
features/output_kmer/kmer_counts/7/
```

Each file is named `<accession>.npy` and has shape `(4**k,)`.

The data loader converts the k-mer counts to relative frequencies.

Feature generation overwrites matching output files.

### 4. Validate feature files

```bash
python cann.py \
    --mode validate \
    --fasta datasets/sequences.fasta \
    --out_root features
```

Resolve any missing or malformed feature files before training.

### 5. Run one cross-validation seed

```bash
python 5CV.py --seed-index 0
```

Each seed index runs all five outer folds.

Indices `0` through `29` correspond to seeds `1` through `30` in `config.py`.

Within each outer fold:

1. The development data are divided into training and validation subsets.
2. Facet normalization is fitted on the training subset.
3. CNN and Transformer encoders are trained independently.
4. Validation performance is used for checkpoint selection.
5. The training and validation samples form the reference collection.
6. The held-out test fold is evaluated.

The test fold is excluded from encoder training, normalization fitting, and checkpoint selection.

### 6. Run all 30 seeds

```bash
for seed_index in {0..29}; do
    python 5CV.py --seed-index "$seed_index" || exit 1
done
```

Each seed writes its results to `RESULT_DIR`.

Use a fresh output directory for a new experiment. Repeating a seed overwrites its corresponding result file.

### 7. Aggregate results

After all 30 seeds finish:

```bash
python 5CV.py --aggregate
```

The final summary is written to:

```text
outputs/final_results.txt
```

If `RESULT_DIR` was changed, the summary is written there instead.

## Outputs

Per-seed results are saved as:

```text
seed_01.txt
...
seed_30.txt
```

The configured methods are:

- CNN 1-NN
- CNN 5-NN
- Transformer 1-NN
- Transformer 5-NN
- Consensus 1-NN
- Consensus 5-NN

Reported metrics include:

- Accuracy
- Balanced accuracy
- Macro-F1
- Macro-recall
- Macro-precision

Balanced accuracy and macro-recall are equivalent for this multiclass evaluation.

The primary summary reports the mean and sample standard deviation of the pooled held-out metrics across 30 seeds.

For reproduction, retain the original dataset, preprocessing, feature settings, model settings, seeds, and software environment. Numerical results can vary across computing environments.

## Acknowledgments

The supplied `psrt.py` acknowledges
[KmerTopology](https://github.com/hozumiyu/KmerTopology)
by Yuta Hozumi. Preserve the applicable attribution for reused code.
