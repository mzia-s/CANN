# CANN

Alignment-free viral family classification using commutative-algebraic descriptors and neural representation learning.

## Table of Contents

- [Introduction](#introduction)
- [CANN Workflow](#cann-workflow)
- [Feature Representation](#feature-representation)
- [Software Requirements](#software-requirements)
- [Repository Organization](#repository-organization)
- [Data Sources and Availability](#data-sources-and-availability)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Evaluation Protocol](#evaluation-protocol)
- [Outputs](#outputs)
- [Acknowledgments](#acknowledgments)

## Introduction

Viral family classification requires sequence representations that capture both nucleotide composition and the positional organization of genomic patterns. CANN combines persistent facet-number descriptors for k-mers of lengths 3, 4, and 5 with complementary k-mer frequency features for lengths 6 and 7.

The five feature blocks are processed by independently trained convolutional neural network (CNN) and Transformer encoders. Cosine distances between the resulting genome embeddings are averaged with equal weights to obtain consensus distances for nearest-neighbor classification. Consensus 1-NN is the principal prediction method, while 5-NN provides an additional evaluation.

The framework builds on the commutative-algebraic sequence representations introduced in [CAKR: commutative algebra k-mer representations for genomics](https://doi.org/10.1038/s41467-026-76429-z), extending their use through supervised neural representation learning and consensus genome comparison.

The main benchmarks use stratified five-fold cross-validation repeated over 30 random seeds. Held-out predictions are pooled across the five folds within each seed, and performance is reported as the mean and sample standard deviation across seeds.

## CANN Workflow

![Overview of the CANN workflow](figures/CANN_workflow.png)

**Figure:** Overview of CANN, including sequence feature generation, independent CNN and Transformer training, consensus distance construction, and nearest-neighbor classification under repeated stratified five-fold cross-validation.

## Feature Representation

Each genome is represented by five feature blocks.

For k = 3, 4, and 5, persistent facet-number descriptors are evaluated at two filtration values: 0 and 4^k. There are 4^k possible nucleotide k-mers, giving two scalar descriptors per k-mer.

For k = 6 and 7, conventional k-mer counts are generated and converted to relative frequencies by the data loader.

| Feature block | k | Number of possible k-mers | Values per k-mer | Scalar features per genome |
| --- | ---: | ---: | ---: | ---: |
| Persistent facet descriptors | 3 | 64 | 2 | 128 |
| Persistent facet descriptors | 4 | 256 | 2 | 512 |
| Persistent facet descriptors | 5 | 1,024 | 2 | 2,048 |
| k-mer frequencies | 6 | 4,096 | 1 | 4,096 |
| k-mer frequencies | 7 | 16,384 | 1 | 16,384 |
| **Total** | | | | **23,168** |

The facet matrices have shapes `(64, 2)`, `(256, 2)`, and `(1024, 2)`. The k-mer composition vectors have lengths 4,096 and 16,384.

Facet normalization is fitted separately within each cross-validation fold using only the training subset. The same transformation is applied to validation, reference, and test samples.

## Software Requirements

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

The recorded PyTorch build uses the CPU. CUDA and a GPU are not required for the commands below.

## Repository Organization

| Location | Contents |
| --- | --- |
| `src/` | Feature generation, data processing, encoders, training, and evaluation code |
| `datasets/NCBI2020/` | NCBI 2020 metadata and corresponding FASTA |
| `datasets/NCBI2022/` | NCBI 2022 metadata and corresponding FASTA |
| `datasets/NCBI2024/` | NCBI 2024 metadata and corresponding FASTA |
| `datasets/NCBI2024_All/` | NCBI 2024 All metadata and corresponding FASTA |
| `figures/` | CANN workflow figure |
| `requirements.txt` | Pinned Python package versions |
| `README.md` | Installation and reproduction instructions |

The following directories are created locally during execution:

| Generated location | Contents |
| --- | --- |
| `features/<dataset>/` | Feature arrays |
| `outputs/<dataset>/` | Per-seed results and aggregate summary |

### Source Files

| File | Purpose |
| --- | --- |
| `src/cann.py` | Count usable genomes, generate features, and validate feature files |
| `src/psrt.py` | Persistent descriptor calculations |
| `src/kmer_only_encoder.py` | Nucleotide k-mer counting |
| `src/config.py` | Paths, feature settings, model settings, and cross-validation configuration |
| `src/data.py` | Sequence and metadata preprocessing, feature loading, and normalization |
| `src/dataset.py` | Datasets, balanced sampling, and data loaders |
| `src/cnn.py` | CNN encoder |
| `src/transformer.py` | Transformer encoder |
| `src/training.py` | Encoder training and validation |
| `src/evaluation.py` | Individual-encoder and consensus evaluation |
| `src/5CV.py` | Repeated stratified five-fold cross-validation |
| `src/results.py` | Result aggregation |

Keep all Python modules together inside `src/`.

## Data Sources and Availability

The genomic data supporting this study were obtained from the National Center for Biotechnology Information (NCBI), including GenBank and, where applicable, the NCBI Virus resource.

Sequence records can be accessed and downloaded through:

- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
- [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/)

The metadata files identify the accession versions and viral family labels used for each dataset. Use the listed accession versions and the supplied family labels when reconstructing the benchmarks, because database sequences and taxonomic assignments may change over time.

### Dataset Files

| Dataset | Metadata CSV | Corresponding FASTA |
| --- | --- | --- |
| NCBI 2020 | `Yau2020_record_processed.csv` | `Yau2020_record_processed.fasta` |
| NCBI 2022 | `Yau2022_record_processed.csv` | `Yau2022_record_processed.fasta` |
| NCBI 2024 | `NCBI_record_valid_nucleotide.csv` | `NCBI_record_valid_nucleotide.fasta` |
| NCBI 2024 All | `NCBI_record_valid_count.csv` | `NCBI_record_valid_count.fasta` |

Place each FASTA file beside its corresponding CSV in the appropriate subfolder of `datasets/`.

The FASTA files are not included in the current repository upload because of their size. Obtain the corresponding sequences before running the workflow. For exact benchmark reproduction, use the original processed FASTA files when available.

### Metadata Columns

All four CSV files contain:

- `Accession (version)`: genome accession including its version.
- `Accession`: accession without the version suffix.
- `Family`: viral family label used for classification.

The NCBI 2020 and NCBI 2022 files additionally contain `Old Family`. The NCBI 2022 file also contains `Baltimore`. The classification target is always `Family`.

### Dataset Sizes Before Pipeline Filtering

| Dataset | Genome records | Viral families |
| --- | ---: | ---: |
| NCBI 2020 | 6,993 | 83 |
| NCBI 2022 | 11,428 | 123 |
| NCBI 2024 | 12,154 | 199 |
| NCBI 2024 All | 13,645 | 209 |

These counts describe the supplied metadata files before pipeline filtering. The code performs sequence cleaning, duplicate handling, and family-size filtering. The main benchmark configuration retains families with at least 15 genomes after preprocessing.

Both the CSV and FASTA are required. The CSV supplies metadata and labels; the FASTA supplies genomic sequences.

## Installation

Download the repository and open a terminal in its main directory.

Create and activate an environment using Python 3.11.3:

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

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The `requirements.txt` file contains:

```text
numpy==1.26.4
scipy==1.17.1
pandas==2.3.0
scikit-learn==1.7.0
torch==2.1.2+cpu
biopython==1.86
gudhi==3.11.0
```

Run all subsequent commands from the main repository directory with the environment activated.

## Reproducing the Experiments

### Step 1: Select and Configure the Dataset

Use the following path section in `src/config.py`. Preserve the remaining feature, model, and training settings.

```python
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Options: NCBI2020, NCBI2022, NCBI2024, NCBI2024_All
DATASET_NAME = "NCBI2022"

DATASET_FILENAMES = {
    "NCBI2020": "Yau2020_record_processed",
    "NCBI2022": "Yau2022_record_processed",
    "NCBI2024": "NCBI_record_valid_nucleotide",
    "NCBI2024_All": "NCBI_record_valid_count",
}

BASE_DIR = PROJECT_DIR / "datasets" / DATASET_NAME
FILE_STEM = DATASET_FILENAMES[DATASET_NAME]

CSV_PATH = BASE_DIR / f"{FILE_STEM}.csv"
FASTA_PATH = BASE_DIR / f"{FILE_STEM}.fasta"

PSRT_BASE = PROJECT_DIR / "features" / DATASET_NAME
KMER_COUNT_ROOT = PSRT_BASE / "output_kmer" / "kmer_counts"

RESULT_DIR = PROJECT_DIR / "outputs" / DATASET_NAME
```

Change `DATASET_NAME` to select the benchmark.

Defining `FASTA_PATH` here overrides the original fallback path in `data.py`. The feature and result directories are separate for each dataset.

### Step 2: Read the Selected Paths

In a Bash terminal, read the configured paths:

```bash
FASTA_PATH=$(python -c \
    "from src.config import FASTA_PATH; print(FASTA_PATH)")

FEATURE_ROOT=$(python -c \
    "from src.config import PSRT_BASE; print(PSRT_BASE)")

printf 'FASTA: %s\nFeatures: %s\n' "$FASTA_PATH" "$FEATURE_ROOT"
```

Use the same terminal for the following feature-generation steps. Repeat this step whenever you change `DATASET_NAME`.

### Step 3: Count Usable Genomes

```bash
python src/cann.py \
    --mode count \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT"
```

The output contains:

```text
USABLE_GENOMES=N
```

Here, `N` is the number of cleaned unique genomes available for feature generation. Worker indices range from `0` through `N - 1`.

### Step 4: Generate Features

To generate features for the first cleaned unique genome:

```bash
python src/cann.py \
    --mode worker \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT" \
    --task_id 0
```

To generate features for every genome sequentially:

```bash
set -euo pipefail

count_output=$(python src/cann.py \
    --mode count \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT")

genome_count=$(printf '%s\n' "$count_output" \
    | awk -F= '/^USABLE_GENOMES=/{print $2}')

if [[ ! "$genome_count" =~ ^[0-9]+$ ]] || (( genome_count == 0 )); then
    echo "A positive usable-genome count is required." >&2
    exit 1
fi

for ((task_id = 0; task_id < genome_count; task_id++)); do
    python src/cann.py \
        --mode worker \
        --fasta "$FASTA_PATH" \
        --out_root "$FEATURE_ROOT" \
        --task_id "$task_id"
done
```

The loop includes task `0`, so running it after the single-genome example regenerates that genome's features.

Each worker reloads the FASTA and processes one genome. Full-dataset generation can take substantial time. Independent task indices can also be scheduled on a computing cluster.

Matching feature files are overwritten when generation is repeated.

### Step 5: Validate Generated Features

After all workers finish:

```bash
python src/cann.py \
    --mode validate \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT"
```

Resolve any missing or malformed feature files before training.

The generated feature locations are:

| Relative location beneath the feature root | Contents |
| --- | --- |
| `output_psrt/all_features/k3/` | Facet arrays of shape `(64, 2)` |
| `output_psrt/all_features/k4/` | Facet arrays of shape `(256, 2)` |
| `output_psrt/all_features/k5/` | Facet arrays of shape `(1024, 2)` |
| `output_kmer/kmer_counts/6/` | Count vectors of length 4,096 |
| `output_kmer/kmer_counts/7/` | Count vectors of length 16,384 |

Facet filenames follow `<accession>_facet0.npy`. K-mer count filenames follow `<accession>.npy`.

### Step 6: Run One Cross-Validation Seed

```bash
python src/5CV.py --seed-index 0
```

This runs all five outer folds for the first configured seed.

Seed indices are zero-based:

- `--seed-index 0` uses seed 1.
- `--seed-index 1` uses seed 2.
- `--seed-index 29` uses seed 30.

Results are written to the configured `RESULT_DIR`. Progress is redirected to the corresponding per-seed text file.

### Step 7: Run All 30 Seeds

```bash
for seed_index in {0..29}; do
    python src/5CV.py --seed-index "$seed_index" || exit 1
done
```

This runs 30 repetitions of five-fold cross-validation, totaling 150 outer-fold evaluations per dataset. Each outer fold trains the CNN and Transformer independently.

The loop includes the first seed. If it was already run in Step 6, its result file will be overwritten.

### Step 8: Aggregate Results

After all 30 seeds finish successfully:

```bash
python src/5CV.py --aggregate
```

For `DATASET_NAME = "NCBI2022"`, the final summary is saved to:

```text
outputs/NCBI2022/final_results.txt
```

The aggregation command expects all configured seed-result files.

### Step 9: Repeat for the Other Datasets

Change `DATASET_NAME` in `src/config.py` and repeat Steps 2–8.

The available dataset identifiers are:

```text
NCBI2020
NCBI2022
NCBI2024
NCBI2024_All
```

Keep each dataset's features and results in its own directory.

## Evaluation Protocol

The default configuration uses:

| Setting | Value |
| --- | --- |
| Outer cross-validation | Stratified five-fold |
| Repetitions | 30 seeds |
| Seed values | 1–30 |
| Minimum family size | 15 genomes |
| Validation fraction of the development set | 0.125 |
| CNN consensus weight | 0.50 |
| Transformer consensus weight | 0.50 |
| Principal classifier | Consensus 1-NN |
| Additional classifier | Consensus 5-NN |

Within each outer fold:

1. The development samples are divided into training and validation subsets.
2. Facet normalization is fitted using only the training subset.
3. CNN and Transformer encoders are trained independently.
4. Validation performance is used for checkpoint selection.
5. Training and validation samples form the reference collection.
6. Held-out test genomes are classified using distances to that reference collection.

The held-out test fold is excluded from encoder training, normalization fitting, and checkpoint selection.

K-mer count vectors are normalized per genome to obtain relative frequencies. Facet features use training-derived min-max normalization and are clipped to the interval `[0, 1]`.

For each seed, held-out predictions are pooled across the five folds before calculating the seed-level metrics. The primary summary reports the mean and sample standard deviation across the 30 seed-level results.

## Outputs

Each dataset produces 30 per-seed result files:

```text
seed_01.txt
seed_02.txt
...
seed_30.txt
```

Aggregation produces:

```text
final_results.txt
```

The configured methods are:

- CNN 1-NN
- CNN 5-NN
- Transformer 1-NN
- Transformer 5-NN
- Consensus 1-NN
- Consensus 5-NN

The reported metrics are:

| Metric | Description |
| --- | --- |
| ACC | Overall accuracy |
| BA | Balanced accuracy |
| F1 | Macro-averaged F1 |
| Recall | Macro-averaged recall |
| Precision | Macro-averaged precision |

Balanced accuracy and macro-recall are equivalent for this multiclass evaluation.

Use the original dataset snapshots, family labels, preprocessing, model settings, seeds, and software environment when reproducing the benchmarks. Numerical results can vary across computing environments.

This workflow covers the main four-dataset cross-validation experiments. The separate NCBI 2026 evaluations and feature-ablation experiments require their corresponding scripts.

## Acknowledgments

We acknowledge Dr. Faisal Suwayyid and collaborators for developing CAKR and making its implementation publicly available. Their work on commutative-algebraic k-mer representations provides a foundation for the algebraic sequence descriptors used in CANN.

- **Paper:** Suwayyid, F., Hozumi, Y., Zia, M., Wee, J., Feng, H., and Wei, G.-W. *CAKR: commutative algebra k-mer representations for genomics*. Nature Communications **17**, 9644 (2026).
- **DOI:** https://doi.org/10.1038/s41467-026-76429-z
- **Code:** https://github.com/FaisalSuwayyid/CAKL

The repository retains the earlier directory name `CAKL`; the published framework is named **CAKR**.

We also retain the acknowledgment of [KmerTopology](https://github.com/hozumiyu/KmerTopology) by Yuta Hozumi included in the supplied `psrt.py`.
