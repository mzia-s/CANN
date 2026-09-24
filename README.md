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
- [Reproducing the Main Benchmarks](#reproducing-the-main-benchmarks)
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

| Feature block | k | Possible k-mers | Values per k-mer | Scalar features |
| --- | ---: | ---: | ---: | ---: |
| Persistent facet descriptors | 3 | 64 | 2 | 128 |
| Persistent facet descriptors | 4 | 256 | 2 | 512 |
| Persistent facet descriptors | 5 | 1,024 | 2 | 2,048 |
| k-mer frequencies | 6 | 4,096 | 1 | 4,096 |
| k-mer frequencies | 7 | 16,384 | 1 | 16,384 |
| **Total per genome** | | | | **23,168** |

The facet matrices have shapes `(64, 2)`, `(256, 2)`, and `(1024, 2)`. The k-mer composition vectors have lengths 4,096 and 16,384.

Facet normalization is fitted separately within each cross-validation fold using only the training subset.

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

The recorded PyTorch build uses the CPU. CUDA and a GPU are not required.

All instructions below use Python directly. Slurm and shell submission scripts are not required.

## Repository Organization

| Location | Contents |
| --- | --- |
| `src/` | Feature generation, preprocessing, models, training, and evaluation code |
| `datasets/` | CSV metadata for the four main benchmarks and the two NCBI 2026 datasets |
| `figures/` | CANN workflow figure |
| `requirements.txt` | Pinned Python package versions |
| `README.md` | Installation and reproduction instructions |

All dataset CSVs are stored directly in `datasets/`, without dataset-specific subfolders. Downloaded FASTA files should be placed in the same directory.

Generated features and results are stored separately for each dataset:

- `features/<dataset>/`
- `outputs/<dataset>/`

### Source Files

| File | Purpose |
| --- | --- |
| `src/cann.py` | Count usable genomes, generate features, and validate feature files |
| `src/psrt.py` | Persistent descriptor calculations |
| `src/kmer_only_encoder.py` | Nucleotide k-mer counting |
| `src/config.py` | Paths, feature settings, model settings, and cross-validation configuration |
| `src/data.py` | Data cleaning, feature loading, and normalization |
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

- [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/)
- [NCBI Nucleotide](https://www.ncbi.nlm.nih.gov/nuccore/)
- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)

### Archived Benchmark Data

The curated data archive associated with the CAKR benchmark study is available on Zenodo:

**[Data for CAKR: commutative algebra k-mer representations for genomics](https://doi.org/10.5281/zenodo.18757928)**

Use the corresponding viral benchmark files from this archive and retain the original filenames listed below.

### Main Benchmark Files

| Dataset | Metadata CSV | Corresponding FASTA |
| --- | --- | --- |
| NCBI 2020 | `Yau2020_record_processed.csv` | `Yau2020_record_processed.fasta` |
| NCBI 2022 | `Yau2022_record_processed.csv` | `Yau2022_record_processed.fasta` |
| NCBI 2024 | `NCBI_record_valid_nucleotide.csv` | `NCBI_record_valid_nucleotide.fasta` |
| NCBI 2024 All | `NCBI_record_valid_count.csv` | `NCBI_record_valid_count.fasta` |

For example, the NCBI 2022 files should be located at:

```text
datasets/Yau2022_record_processed.csv
datasets/Yau2022_record_processed.fasta
```

The metadata files contain the following numbers of records before pipeline filtering:

| Dataset | Genome records | Viral families |
| --- | ---: | ---: |
| NCBI 2020 | 6,993 | 83 |
| NCBI 2022 | 11,428 | 123 |
| NCBI 2024 | 12,154 | 199 |
| NCBI 2024 All | 13,645 | 209 |

The main classification pipeline performs sequence cleaning, duplicate handling, and family-size filtering. Families with at least 15 genomes after preprocessing are retained.

### NCBI 2026 and NCBI 2026 All

The CSV metadata for NCBI 2026 and NCBI 2026 All are also provided directly in `datasets/`.

Their corresponding genomic sequences can be downloaded from [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/) or [NCBI Nucleotide](https://www.ncbi.nlm.nih.gov/nuccore/) using the accession.version identifiers in the CSV files. The processed FASTA files used in this study can also be provided by the authors upon request.

The 2026 evaluations use distinct reference-coverage and evaluation protocols. The main-benchmark commands below do not reproduce those experiments simply by selecting a 2026 CSV; their corresponding evaluation scripts and settings are required.

### Obtaining and Preparing FASTA Files

FASTA files are not included in this GitHub repository because of their size.

1. Obtain the corresponding benchmark sequences from the linked archive, download them from NCBI, or request the processed FASTA files from the authors.
2. When downloading from NCBI, use the exact accession.version identifiers listed in the dataset CSV.
3. Save the sequences in FASTA format.
4. Place the FASTA directly in `datasets/`, beside its CSV.
5. Use the same filename stem as the CSV, replacing `.csv` with `.fasta`.

Retain the family labels supplied in the CSV files. Current NCBI taxonomic assignments may differ from those used in the original benchmarks.

For exact reproduction, use the original processed sequence files whenever possible.

### Metadata Columns

The four main benchmark CSVs contain:

- `Accession (version)`: genome accession including its version.
- `Accession`: accession without the version suffix.
- `Family`: viral family label used for classification.

The NCBI 2020 and NCBI 2022 metadata additionally contain `Old Family`. NCBI 2022 also contains `Baltimore`.

The classification target is `Family`.

Both CSV metadata and FASTA sequences are required to run the supplied workflow.

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

Run all subsequent commands from the main repository directory with this environment activated.

## Reproducing the Main Benchmarks

The following commands run the complete feature-generation and repeated cross-validation workflow directly, without Slurm submission scripts.

### Step 1: Configure the Dataset

Use the following path section in `src/config.py`, preserving the remaining feature, model, and training settings:

```python
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Main benchmark options:
# NCBI2020, NCBI2022, NCBI2024, NCBI2024_All
DATASET_NAME = "NCBI2022"

DATASET_FILENAMES = {
    "NCBI2020": "Yau2020_record_processed",
    "NCBI2022": "Yau2022_record_processed",
    "NCBI2024": "NCBI_record_valid_nucleotide",
    "NCBI2024_All": "NCBI_record_valid_count",
}

BASE_DIR = PROJECT_DIR / "datasets"
FILE_STEM = DATASET_FILENAMES[DATASET_NAME]

CSV_PATH = BASE_DIR / f"{FILE_STEM}.csv"
FASTA_PATH = BASE_DIR / f"{FILE_STEM}.fasta"

PSRT_BASE = PROJECT_DIR / "features" / DATASET_NAME
KMER_COUNT_ROOT = PSRT_BASE / "output_kmer" / "kmer_counts"

RESULT_DIR = PROJECT_DIR / "outputs" / DATASET_NAME
```

Only the four main benchmarks are included in this configuration mapping.

Defining `FASTA_PATH` here overrides the original fallback path in `data.py`.

### Step 2: Read and Check the Selected Paths

Run the following in Bash:

```bash
FASTA_PATH=$(python -c \
    "from src.config import FASTA_PATH; print(FASTA_PATH)")

FEATURE_ROOT=$(python -c \
    "from src.config import PSRT_BASE; print(PSRT_BASE)")

python - <<'PY'
from src.config import CSV_PATH, FASTA_PATH

for path in (CSV_PATH, FASTA_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    print(f"Found: {path}")
PY

printf 'Feature directory: %s\n' "$FEATURE_ROOT"
```

Use the same terminal for the following feature-generation steps. Repeat this step after changing `DATASET_NAME`.

### Step 3: Count Usable Genomes

```bash
python src/cann.py \
    --mode count \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT"
```

The output includes:

```text
USABLE_GENOMES=N
```

Here, `N` is the number of cleaned unique genomes available for feature generation. Worker indices range from `0` through `N - 1`.

### Step 4: Generate Features for All Genomes

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

Each worker reloads the FASTA and processes one genome. This loop runs sequentially and may take substantial time for a full dataset.

To process a single genome, use its task index:

```bash
python src/cann.py \
    --mode worker \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT" \
    --task_id 0
```

Matching feature files are overwritten when generation is repeated.

### Step 5: Validate Generated Features

After all genomes have been processed:

```bash
python src/cann.py \
    --mode validate \
    --fasta "$FASTA_PATH" \
    --out_root "$FEATURE_ROOT"
```

Resolve missing or malformed feature files before training.

The generated files are organized beneath the selected feature root:

| Relative directory | Feature shape per genome |
| --- | --- |
| `output_psrt/all_features/k3/` | `(64, 2)` |
| `output_psrt/all_features/k4/` | `(256, 2)` |
| `output_psrt/all_features/k5/` | `(1024, 2)` |
| `output_kmer/kmer_counts/6/` | `(4096,)` |
| `output_kmer/kmer_counts/7/` | `(16384,)` |

Facet filenames follow `<accession>_facet0.npy`. K-mer count filenames follow `<accession>.npy`.

### Step 6: Set the CPU Thread Configuration

Use the thread settings from the original cross-validation launcher:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Set these variables before starting the training processes.

### Step 7: Run Cross-Validation

To run one seed containing all five outer folds:

```bash
python src/5CV.py --seed-index 0
```

Seed indices are zero-based:

- Index `0` uses seed 1.
- Index `1` uses seed 2.
- Index `29` uses seed 30.

To run all 30 seeds sequentially:

```bash
for seed_index in {0..29}; do
    python src/5CV.py --seed-index "$seed_index" || exit 1
done
```

This performs 150 outer-fold evaluations per dataset. Each fold trains the CNN and Transformer independently.

The full loop includes seed index `0`. If that seed was already run separately, its output is overwritten.

Progress and results are redirected to per-seed text files in `RESULT_DIR`.

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

### Step 9: Repeat for the Other Main Benchmarks

Change `DATASET_NAME` in `src/config.py` and repeat Steps 2–8.

Available selections:

```text
NCBI2020
NCBI2022
NCBI2024
NCBI2024_All
```

CSV and FASTA inputs remain directly inside `datasets/`. Generated features and results are separated by dataset.

## Evaluation Protocol

The default main-benchmark configuration uses:

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

1. Development samples are divided into training and validation subsets.
2. Facet normalization is fitted using only the training subset.
3. CNN and Transformer encoders are trained independently.
4. Validation performance is used for checkpoint selection.
5. Training and validation samples form the reference collection.
6. Held-out test genomes are classified using distances to the reference collection.

The held-out test fold is excluded from encoder training, normalization fitting, and checkpoint selection.

K-mer count vectors are normalized per genome to obtain relative frequencies. Facet features use training-derived min-max normalization and are clipped to `[0, 1]`.

For each seed, held-out predictions are pooled across the five folds before calculating seed-level metrics. The primary summary reports the mean and sample standard deviation across the 30 seed-level results.

## Outputs

Each main benchmark produces:

- `seed_01.txt` through `seed_30.txt`
- `final_results.txt`

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

For reproduction, retain the original sequence snapshots, family labels, preprocessing, feature settings, model settings, seeds, and software environment. Numerical results can vary across computing environments.

The commands above cover the four main cross-validation benchmarks. Separate NCBI 2026 evaluations and feature-ablation experiments require their corresponding evaluation scripts and settings.

## Acknowledgments

We acknowledge Dr. Faisal Suwayyid and collaborators for developing CAKR and making its implementation publicly available. Their work on commutative-algebraic k-mer representations provides a foundation for the algebraic sequence descriptors used in CANN.

- **Paper:** Suwayyid, F., Hozumi, Y., Zia, M., Wee, J., Feng, H., and Wei, G.-W. *CAKR: commutative algebra k-mer representations for genomics*. Nature Communications **17**, 9644 (2026).
- **DOI:** https://doi.org/10.1038/s41467-026-76429-z
- **Code:** https://github.com/FaisalSuwayyid/CAKL
- **Data archive:** https://doi.org/10.5281/zenodo.18757928

The repository retains the earlier directory name `CAKL`; the published framework is named **CAKR**.

We also retain the acknowledgment of [KmerTopology](https://github.com/hozumiyu/KmerTopology) by Yuta Hozumi included in the supplied `psrt.py`.
