#!/usr/bin/env python3

from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    "/mnt/home/ziamusha/Phylogenetics/NCBI"
)

CSV_PATH = (
    BASE_DIR
    / "Yau2022_record_processed.csv"
)

PSRT_BASE = Path(
    "/mnt/gs21/scratch/ziamusha/DeepLearning/Yau2022_features"
)

KMER_COUNT_ROOT = Path(
    "/mnt/gs21/scratch/ziamusha/DeepLearning/"
    "Yau2022_features/output_kmer/kmer_counts"
)

RESULT_DIR = Path(
    "/mnt/home/ziamusha/Phylogenetics/DeepLearning/"
    "5CV_NCBI2022_results"
)

# ============================================================
# 2. REPEATED 5-FOLD CV SETTINGS
# ============================================================

CV_SEEDS = list(range(1, 31))

NUMBER_OF_FOLDS = 5

VALIDATION_FRACTION_OF_DEVELOPMENT = 0.125

MINIMUM_FAMILY_SIZE = 15

# ============================================================
# 3. FINAL FEATURE REPRESENTATION
# ============================================================

FACET_K_VALUES = [
    3,
    4,
    5,
]

KMER_K_VALUES = [
    6,
    7,
]

INPUT_SPECS = [
    ("facet", 3),
    ("facet", 4),
    ("facet", 5),
    ("kmer", 6),
    ("kmer", 7),
]

FACET_FEATURE_NAME = "facet0"

NUMBER_OF_FILTRATION_POSITIONS = 2

# ============================================================
# 4. FIXED CONSENSUS
# ============================================================

CNN_CONSENSUS_WEIGHT = 0.50

TRANSFORMER_CONSENSUS_WEIGHT = 0.50

# ============================================================
# 5. CNN SETTINGS
# ============================================================

CNN_ENCODER_DIMENSION = 128

CNN_MAXIMUM_EPOCHS = 60

CNN_LEARNING_RATE = 0.0005

CNN_WEIGHT_DECAY = 0.0001

CNN_BRANCH_DROPOUT = 0.10

# ============================================================
# 6. TRANSFORMER SETTINGS
# ============================================================

TRANSFORMER_DIMENSION = 128

TRANSFORMER_LAYERS = 2

TRANSFORMER_HEADS = 4

TRANSFORMER_DROPOUT = 0.10

TRANSFORMER_MAXIMUM_EPOCHS = 60

TRANSFORMER_LEARNING_RATE = 0.0005

TRANSFORMER_WEIGHT_DECAY = 0.0001

# ============================================================
# 7. REPRESENTATION LEARNING
# ============================================================

PROJECTION_DIMENSION = 128

CONTRASTIVE_WEIGHT = 0.10

CONTRASTIVE_TEMPERATURE = 0.07

# ============================================================
# 8. OPTIMIZATION
# ============================================================

GRADIENT_CLIP_NORM = 5.0

EARLY_STOPPING_PATIENCE = 10

SCHEDULER_PATIENCE = 3

SCHEDULER_FACTOR = 0.5

MINIMUM_LEARNING_RATE = 1e-6

CHECKPOINT_TOLERANCE = 1e-8

# ============================================================
# 9. BALANCED BATCH SETTINGS
# ============================================================

MAX_FAMILIES_PER_BATCH = 16

GENOMES_PER_FAMILY = 4

EVALUATION_BATCH_SIZE = 128

NUM_WORKERS = 0

# ============================================================
# 10. CHECKPOINT SELECTION
# ============================================================

PRIMARY_SELECTION_METHOD = "Encoder + Cosine 1-NN"

# ============================================================
# 11. FINAL METHODS
# ============================================================

METHODS = [
    "CNN 1-NN",
    "CNN 5-NN",
    "Transformer 1-NN",
    "Transformer 5-NN",
    "Consensus 1-NN",
    "Consensus 5-NN",
]

METRIC_NAMES = [
    "ACC",
    "BA",
    "F1",
    "Recall",
    "Precision",
]