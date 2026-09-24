#!/usr/bin/env python3

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    CSV_PATH,
    PSRT_BASE,
    KMER_COUNT_ROOT,
    MINIMUM_FAMILY_SIZE,
    FACET_K_VALUES,
    KMER_K_VALUES,
    FACET_FEATURE_NAME,
    NUMBER_OF_FILTRATION_POSITIONS,
)


# ============================================================
# TARGET COLUMN
#
# IMPORTANT:
# Always classify viral FAMILY.
# Do not infer the target from the last CSV column because
# Yau2022 contains an additional "Baltimore" column.
# ============================================================

FAMILY_COLUMN = "Family"


# ============================================================
# FASTA PATH
# ============================================================
#
# Use FASTA_PATH from config.py if it is defined there.
# Otherwise use the Yau2022 processed FASTA directly.
# ============================================================

try:
    from config import FASTA_PATH

except ImportError:

    FASTA_PATH = Path(
        "/mnt/home/*************/NCBI/"
        "Yau2022_record_processed.fasta"
    )


CSV_PATH = Path(CSV_PATH)
PSRT_BASE = Path(PSRT_BASE)
KMER_COUNT_ROOT = Path(KMER_COUNT_ROOT)
FASTA_PATH = Path(FASTA_PATH)

VALID_DNA_CHARACTERS = frozenset("ACGT")


# ============================================================
# PRINT HELPER
# ============================================================

def progress_print(*args):
    print(*args, flush=True)


# ============================================================
# ACCESSION EXTRACTION
# ============================================================

def extract_accession(text):

    match = re.search(
        r"([A-Z]{1,5}_?\d+\.\d+)",
        str(text),
    )

    return (
        None
        if match is None
        else match.group(1)
    )


# ============================================================
# FASTA READER
# ============================================================

def read_fasta_records(fasta_path):
    """
    Yield (header, sequence) records without requiring Biopython.
    """

    header = None
    sequence_parts = []

    with fasta_path.open("r") as handle:

        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:

                    yield (
                        header,
                        "".join(sequence_parts),
                    )

                header = line[1:].strip()
                sequence_parts = []

                continue

            if header is None:

                raise ValueError(
                    "FASTA sequence encountered before "
                    "the first header.\n"
                    f"File: {fasta_path}\n"
                    f"Line: {line_number}"
                )

            sequence_parts.append(
                line
            )

    if header is not None:

        yield (
            header,
            "".join(sequence_parts),
        )


# ============================================================
# CSV CLEANING
# ============================================================

def clean_metadata(
    df,
    accession_column,
    family_column,
):
    """
    Duplicate CSV accession rules:

    same accession + same family
        -> keep one

    same accession + different family
        -> ERROR
    """

    original_rows = len(df)

    df = df.copy()


    # --------------------------------------------------------
    # Missing accession/family
    # --------------------------------------------------------

    missing_mask = (
        df[accession_column].isna()
        |
        df[family_column].isna()
    )

    missing_rows = int(
        missing_mask.sum()
    )

    if missing_rows:

        df = df.loc[
            ~missing_mask
        ].copy()


    # --------------------------------------------------------
    # Strip whitespace
    # --------------------------------------------------------

    df[accession_column] = (
        df[accession_column]
        .astype(str)
        .str.strip()
    )

    df[family_column] = (
        df[family_column]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Empty strings / string "nan"
    # --------------------------------------------------------

    empty_mask = (

        df[accession_column].eq("")
        |
        df[family_column].eq("")
        |
        df[accession_column]
        .str.lower()
        .eq("nan")
        |
        df[family_column]
        .str.lower()
        .eq("nan")
    )

    empty_rows = int(
        empty_mask.sum()
    )

    if empty_rows:

        df = df.loc[
            ~empty_mask
        ].copy()


    # --------------------------------------------------------
    # Duplicate accession validation
    # --------------------------------------------------------

    duplicate_rows = df.loc[
        df[accession_column].duplicated(
            keep=False
        )
    ].copy()

    if not duplicate_rows.empty:

        family_counts = (
            duplicate_rows
            .groupby(
                accession_column
            )[family_column]
            .nunique()
        )

        conflicting = (
            family_counts[
                family_counts > 1
            ]
            .index
            .tolist()
        )

        if conflicting:

            examples = []

            for accession in conflicting[:20]:

                families = (
                    duplicate_rows.loc[
                        duplicate_rows[
                            accession_column
                        ]
                        ==
                        accession,
                        family_column,
                    ]
                    .drop_duplicates()
                    .tolist()
                )

                examples.append(
                    f"{accession}: {families}"
                )

            raise ValueError(
                "The same accession has different "
                "family labels in the CSV.\n"
                "This cannot be resolved safely.\n"
                "Examples:\n"
                +
                "\n".join(examples)
            )


    # --------------------------------------------------------
    # Keep one copy of duplicate accession rows
    # --------------------------------------------------------

    before = len(df)

    df = (
        df
        .drop_duplicates(
            subset=[accession_column],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    return df, {

        "original_csv_rows":
            original_rows,

        "missing_metadata_rows_removed":
            missing_rows + empty_rows,

        "duplicate_csv_accession_rows_removed":
            before - len(df),
    }


# ============================================================
# FASTA CLEANING
# ============================================================

def read_and_clean_fasta(
    fasta_path,
):
    """
    FASTA rules:

    uppercase

    U -> T

    remove every remaining non-ACGT character

    skip only if cleaned sequence becomes empty

    same accession + same cleaned sequence
        -> keep one

    same accession + different cleaned sequence
        -> ERROR

    Example:
        ACNT -> ACT
    """

    if not fasta_path.exists():

        raise FileNotFoundError(
            f"FASTA not found:\n{fasta_path}"
        )


    sequence_by_accession = {}
    accession_order = []

    total_records = 0
    no_accession_records = 0

    u_converted_sequences = 0
    total_u_letters_converted = 0

    cleaned_nonstandard_sequences = 0
    total_nonstandard_characters_removed = 0

    removed_character_counts = Counter()
    cleaned_examples = []

    empty_after_cleaning = []

    duplicate_same_accession_same_sequence = []

    conflicting_same_accession_sequences = []


    # --------------------------------------------------------
    # Process FASTA
    # --------------------------------------------------------

    for (
        header,
        raw_sequence,
    ) in read_fasta_records(
        fasta_path
    ):

        total_records += 1

        accession = extract_accession(
            header
        )

        if accession is None:

            no_accession_records += 1

            continue


        # ----------------------------------------------------
        # Uppercase
        # ----------------------------------------------------

        sequence = (
            raw_sequence.upper()
        )


        # ----------------------------------------------------
        # U -> T
        # ----------------------------------------------------

        u_count = sequence.count(
            "U"
        )

        if u_count:

            u_converted_sequences += 1

            total_u_letters_converted += (
                u_count
            )

            sequence = sequence.replace(
                "U",
                "T",
            )


        # ----------------------------------------------------
        # Remove non-ACGT
        # ----------------------------------------------------

        removed = Counter(

            character

            for character
            in sequence

            if character
            not in VALID_DNA_CHARACTERS
        )

        if removed:

            cleaned_nonstandard_sequences += 1

            removed_count = sum(
                removed.values()
            )

            total_nonstandard_characters_removed += (
                removed_count
            )

            removed_character_counts.update(
                removed
            )

            old_length = len(
                sequence
            )

            cleaned_sequence = "".join(

                character

                for character
                in sequence

                if character
                in VALID_DNA_CHARACTERS
            )

            if len(cleaned_examples) < 20:

                cleaned_examples.append(
                    (
                        accession,
                        "".join(
                            sorted(
                                removed.keys()
                            )
                        ),
                        removed_count,
                        old_length,
                        len(cleaned_sequence),
                    )
                )

            sequence = (
                cleaned_sequence
            )


        # ----------------------------------------------------
        # Empty sequence
        # ----------------------------------------------------

        if len(sequence) == 0:

            empty_after_cleaning.append(
                accession
            )

            continue


        # ----------------------------------------------------
        # Duplicate accession in FASTA
        # ----------------------------------------------------

        if accession in sequence_by_accession:

            if (
                sequence_by_accession[
                    accession
                ]
                ==
                sequence
            ):

                duplicate_same_accession_same_sequence.append(
                    accession
                )

            else:

                conflicting_same_accession_sequences.append(
                    accession
                )

            continue


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        sequence_by_accession[
            accession
        ] = sequence

        accession_order.append(
            accession
        )


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if total_records == 0:

        raise ValueError(
            f"No FASTA records found in:\n"
            f"{fasta_path}"
        )


    if conflicting_same_accession_sequences:

        conflicts = list(
            dict.fromkeys(
                conflicting_same_accession_sequences
            )
        )

        raise ValueError(
            "The same accession occurs in the FASTA "
            "with different cleaned nucleotide sequences.\n"
            "The loader will not keep both and will not "
            "choose one arbitrarily.\n"
            "These accessions must be resolved before "
            "5-fold CV.\n"
            f"Examples: {conflicts[:20]}"
        )


    return (
        sequence_by_accession,
        accession_order,
        {

            "total_fasta_records":
                total_records,

            "no_accession_records_skipped":
                no_accession_records,

            "u_converted_sequences":
                u_converted_sequences,

            "total_u_letters_converted":
                total_u_letters_converted,

            "cleaned_nonstandard_sequences":
                cleaned_nonstandard_sequences,

            "total_nonstandard_characters_removed":
                total_nonstandard_characters_removed,

            "removed_character_counts":
                dict(
                    sorted(
                        removed_character_counts.items()
                    )
                ),

            "cleaned_examples":
                cleaned_examples,

            "empty_after_cleaning_skipped":
                len(
                    empty_after_cleaning
                ),

            "empty_after_cleaning_examples":
                empty_after_cleaning[:20],

            "duplicate_fasta_accession_records_removed":
                len(
                    duplicate_same_accession_same_sequence
                ),

            "duplicate_fasta_accession_examples":
                duplicate_same_accession_same_sequence[:20],
        },
    )


# ============================================================
# CSV/FASTA ALIGNMENT + EXACT-SEQUENCE DEDUPLICATION
# ============================================================

def align_and_remove_duplicate_sequences(
    df,
    accession_column,
    family_column,
    sequence_by_accession,
    fasta_accession_order,
):
    """
    Different-accession exact-sequence rules:

    different accession + same sequence + same family
        -> keep one

    different accession + same sequence + different family
        -> ERROR
    """

    accession_to_family = dict(
        zip(
            df[accession_column],
            df[family_column],
        )
    )

    csv_accession_set = set(
        accession_to_family
    )

    fasta_accession_set = set(
        sequence_by_accession
    )


    # --------------------------------------------------------
    # Accessions present only on one side
    # --------------------------------------------------------

    csv_without_valid_fasta = sorted(
        csv_accession_set
        -
        fasta_accession_set
    )

    fasta_without_csv = [

        accession

        for accession
        in fasta_accession_order

        if accession
        not in csv_accession_set
    ]


    # --------------------------------------------------------
    # Exact-sequence deduplication
    # --------------------------------------------------------

    sequence_to_representative = {}

    retained_rows = []

    duplicate_sequences_removed = []

    conflicting_sequence_labels = []


    for accession in fasta_accession_order:

        if accession not in accession_to_family:
            continue

        family = accession_to_family[
            accession
        ]

        sequence = sequence_by_accession[
            accession
        ]


        if sequence in sequence_to_representative:

            (
                kept_accession,
                kept_family,
            ) = sequence_to_representative[
                sequence
            ]

            if family != kept_family:

                conflicting_sequence_labels.append(
                    (
                        kept_accession,
                        kept_family,
                        accession,
                        family,
                    )
                )

            else:

                duplicate_sequences_removed.append(
                    (
                        accession,
                        kept_accession,
                        family,
                    )
                )

            continue


        sequence_to_representative[
            sequence
        ] = (
            accession,
            family,
        )

        retained_rows.append(
            {
                accession_column:
                    accession,

                family_column:
                    family,
            }
        )


    # --------------------------------------------------------
    # Sequence label conflict
    # --------------------------------------------------------

    if conflicting_sequence_labels:

        examples = [

            f"{a1} ({f1}) <-> "
            f"{a2} ({f2})"

            for (
                a1,
                f1,
                a2,
                f2,
            )
            in conflicting_sequence_labels[:20]
        ]

        raise ValueError(
            "The exact same FASTA sequence occurs "
            "under different family labels.\n"
            "This is a class-label conflict and cannot "
            "be used safely for 5-fold CV.\n"
            "Examples:\n"
            +
            "\n".join(examples)
        )


    aligned_df = pd.DataFrame(
        retained_rows,
        columns=[
            accession_column,
            family_column,
        ],
    )


    if aligned_df.empty:

        raise ValueError(
            "No genomes remain after matching the CSV "
            "to valid FASTA records."
        )


    return (
        aligned_df,
        {

            "csv_without_valid_fasta":
                len(
                    csv_without_valid_fasta
                ),

            "csv_without_valid_fasta_examples":
                csv_without_valid_fasta[:20],

            "fasta_without_csv":
                len(
                    fasta_without_csv
                ),

            "fasta_without_csv_examples":
                fasta_without_csv[:20],

            "duplicate_sequence_records_removed":
                len(
                    duplicate_sequences_removed
                ),

            "duplicate_sequence_examples":
                duplicate_sequences_removed[:20],
        },
    )


# ============================================================
# FEATURE FILE MAPPING
# ============================================================

def build_feature_mapping(
    directory,
    pattern,
    required_set,
    required_accessions,
    description,
):

    files = list(
        directory.glob(
            pattern
        )
    )


    if not files:

        raise FileNotFoundError(
            f"No {description} files found:\n"
            f"{directory}"
        )


    mapping = {}
    duplicates = []


    for file_path in files:

        accession = extract_accession(
            file_path.name
        )

        if (
            accession is None
            or
            accession not in required_set
        ):

            continue


        if accession in mapping:

            duplicates.append(
                accession
            )

        else:

            mapping[
                accession
            ] = file_path


    if duplicates:

        duplicates = list(
            dict.fromkeys(
                duplicates
            )
        )

        raise ValueError(
            f"Duplicate {description} mappings.\n"
            f"Examples: {duplicates[:20]}"
        )


    missing = [

        accession

        for accession
        in required_accessions

        if accession
        not in mapping
    ]


    if missing:

        raise FileNotFoundError(
            f"{description} incomplete.\n"
            f"Mapped: {len(mapping)} / "
            f"{len(required_accessions)}\n"
            f"Missing examples: {missing[:20]}"
        )


    return (
        files,
        mapping,
    )


# ============================================================
# DIRECTORY / SHAPE HELPERS
# ============================================================

def get_facet_directory(k):

    return (
        PSRT_BASE
        /
        "output_psrt"
        /
        "all_features"
        /
        f"k{k}"
    )


def get_kmer_directory(k):

    return (
        KMER_COUNT_ROOT
        /
        str(k)
    )


def expected_facet_shape(k):

    return (
        4 ** k,
        NUMBER_OF_FILTRATION_POSITIONS,
    )


def expected_kmer_shape(k):

    return (
        4 ** k,
    )


# ============================================================
# LOAD COMPLETE DATASET
# ============================================================

def load_complete_dataset():

    # --------------------------------------------------------
    # Basic path checks
    # --------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )


    if not FASTA_PATH.exists():

        raise FileNotFoundError(
            f"FASTA not found:\n{FASTA_PATH}"
        )


    if not KMER_COUNT_ROOT.exists():

        raise FileNotFoundError(
            f"K-mer root not found:\n"
            f"{KMER_COUNT_ROOT}"
        )


    # ========================================================
    # READ CSV
    # ========================================================

    df_raw = pd.read_csv(
        CSV_PATH
    )


    # --------------------------------------------------------
    # Accession column
    # --------------------------------------------------------

    accession_column = (
        "Accession (version)"
    )

    if accession_column not in df_raw.columns:

        raise KeyError(
            f"Required column "
            f"{accession_column!r} "
            f"was not found in:\n"
            f"{CSV_PATH}\n\n"
            f"Available columns:\n"
            f"{list(df_raw.columns)}"
        )


    # ========================================================
    # IMPORTANT FIX:
    # EXPLICIT FAMILY TARGET
    # ========================================================

    family_column = (
        FAMILY_COLUMN
    )

    if family_column not in df_raw.columns:

        raise KeyError(
            f"Required classification target "
            f"{family_column!r} was not found in:\n"
            f"{CSV_PATH}\n\n"
            f"Available columns:\n"
            f"{list(df_raw.columns)}"
        )


    # --------------------------------------------------------
    # Explicit safety check
    # --------------------------------------------------------

    if family_column != "Family":

        raise RuntimeError(
            "Classification target must be 'Family'. "
            f"Received: {family_column!r}"
        )


    # --------------------------------------------------------
    # Count original viral families
    # --------------------------------------------------------

    original_family_count = (
        df_raw[
            family_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .nunique()
    )


    # --------------------------------------------------------
    # Initial diagnostic output
    # --------------------------------------------------------

    progress_print(
        "=" * 120
    )

    progress_print(
        "TARGET VALIDATION"
    )

    progress_print(
        "=" * 120
    )

    progress_print(
        "CSV columns:",
        list(
            df_raw.columns
        ),
    )

    progress_print(
        "Classification target column:",
        family_column,
    )

    progress_print(
        "Original viral families:",
        original_family_count,
    )

    if "Baltimore" in df_raw.columns:

        baltimore_count = (
            df_raw[
                "Baltimore"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

        progress_print(
            "Baltimore classes present in CSV:",
            baltimore_count,
        )

        progress_print(
            "Baltimore is NOT used as the "
            "classification target."
        )


    # ========================================================
    # 1. CLEAN METADATA
    # ========================================================

    (
        df_metadata,
        metadata_stats,
    ) = clean_metadata(
        df_raw,
        accession_column,
        family_column,
    )


    # ========================================================
    # 2. READ AND CLEAN FASTA
    # ========================================================

    (
        sequence_by_accession,
        fasta_accession_order,
        fasta_stats,
    ) = read_and_clean_fasta(
        FASTA_PATH
    )


    # ========================================================
    # 3. MATCH CSV TO FASTA AND REMOVE EXACT DUPLICATES
    # ========================================================

    (
        df_sequence_cleaned,
        alignment_stats,
    ) = align_and_remove_duplicate_sequences(
        df_metadata,
        accession_column,
        family_column,
        sequence_by_accession,
        fasta_accession_order,
    )


    # ========================================================
    # 4. APPLY MINIMUM FAMILY-SIZE FILTER
    #
    # IMPORTANT:
    # Filtering occurs AFTER sequence cleaning.
    # ========================================================

    family_counts = (
        df_sequence_cleaned[
            family_column
        ]
        .value_counts()
    )


    valid_families = (
        family_counts[
            family_counts
            >=
            MINIMUM_FAMILY_SIZE
        ]
        .index
    )


    df_filtered = (
        df_sequence_cleaned[
            df_sequence_cleaned[
                family_column
            ].isin(
                valid_families
            )
        ]
        .reset_index(
            drop=True
        )
    )


    if df_filtered.empty:

        raise ValueError(
            "No genomes remain after FASTA cleaning, "
            "duplicate removal, and minimum-family-size "
            "filtering."
        )


    # --------------------------------------------------------
    # Required accessions
    # --------------------------------------------------------

    required_accessions = (
        df_filtered[
            accession_column
        ]
        .to_numpy(
            dtype=str
        )
    )


    if (
        len(required_accessions)
        !=
        len(
            set(
                required_accessions
            )
        )
    ):

        raise RuntimeError(
            "Internal error: duplicate accession IDs "
            "remain after cleaning."
        )


    required_accession_set = set(
        required_accessions
    )


    # ========================================================
    # FAMILY LABEL ENCODING
    # ========================================================

    label_encoder = (
        LabelEncoder()
    )


    y = (
        label_encoder
        .fit_transform(
            df_filtered[
                family_column
            ]
        )
        .astype(
            np.int64
        )
    )


    family_names = (
        label_encoder.classes_
    )


    number_of_classes = (
        len(
            family_names
        )
    )


    class_counts = np.bincount(
        y,
        minlength=number_of_classes,
    )


    if (
        class_counts.min()
        <
        MINIMUM_FAMILY_SIZE
    ):

        raise RuntimeError(
            "Internal error: a retained family "
            "has fewer than "
            f"{MINIMUM_FAMILY_SIZE} genomes."
        )


    # ========================================================
    # CLEANING SUMMARY
    # ========================================================

    progress_print(
        "=" * 120
    )

    progress_print(
        "DATASET CLEANING AND VALIDATION"
    )

    progress_print(
        "=" * 120
    )


    progress_print(
        "CSV:",
        CSV_PATH,
    )

    progress_print(
        "FASTA:",
        FASTA_PATH,
    )

    progress_print(
        "Classification target:",
        family_column,
    )

    progress_print(
        "Original CSV genomes:",
        metadata_stats[
            "original_csv_rows"
        ],
    )

    progress_print(
        "Original families:",
        original_family_count,
    )


    progress_print(
        "Metadata rows with missing "
        "accession/family removed:",
        metadata_stats[
            "missing_metadata_rows_removed"
        ],
    )


    progress_print(
        "Duplicate CSV accession rows removed "
        "(same accession + same family):",
        metadata_stats[
            "duplicate_csv_accession_rows_removed"
        ],
    )


    progress_print(
        "Total FASTA records:",
        fasta_stats[
            "total_fasta_records"
        ],
    )


    progress_print(
        "FASTA records without recognizable "
        "accession skipped:",
        fasta_stats[
            "no_accession_records_skipped"
        ],
    )


    progress_print(
        "FASTA sequences containing U "
        "converted to T:",
        fasta_stats[
            "u_converted_sequences"
        ],
    )


    progress_print(
        "Total U letters converted to T:",
        fasta_stats[
            "total_u_letters_converted"
        ],
    )


    progress_print(
        "FASTA sequences cleaned for "
        "nonstandard letters:",
        fasta_stats[
            "cleaned_nonstandard_sequences"
        ],
    )


    progress_print(
        "Total nonstandard letters removed:",
        fasta_stats[
            "total_nonstandard_characters_removed"
        ],
    )


    if fasta_stats[
        "removed_character_counts"
    ]:

        progress_print(
            "Removed-character counts:",
            fasta_stats[
                "removed_character_counts"
            ],
        )


    if fasta_stats[
        "cleaned_examples"
    ]:

        progress_print(
            "Cleaned FASTA examples "
            "(accession, removed characters, "
            "count, old length, new length):",
            fasta_stats[
                "cleaned_examples"
            ],
        )


    progress_print(
        "Empty-after-cleaning FASTA "
        "records skipped:",
        fasta_stats[
            "empty_after_cleaning_skipped"
        ],
    )


    progress_print(
        "Duplicate FASTA records removed "
        "(same accession + same sequence):",
        fasta_stats[
            "duplicate_fasta_accession_records_removed"
        ],
    )


    if fasta_stats[
        "duplicate_fasta_accession_examples"
    ]:

        progress_print(
            "Duplicate FASTA accession examples:",
            fasta_stats[
                "duplicate_fasta_accession_examples"
            ],
        )


    progress_print(
        "CSV accessions without a valid "
        "FASTA sequence skipped:",
        alignment_stats[
            "csv_without_valid_fasta"
        ],
    )


    if alignment_stats[
        "csv_without_valid_fasta_examples"
    ]:

        progress_print(
            "CSV without valid FASTA examples:",
            alignment_stats[
                "csv_without_valid_fasta_examples"
            ],
        )


    progress_print(
        "Valid FASTA accessions absent "
        "from CSV ignored:",
        alignment_stats[
            "fasta_without_csv"
        ],
    )


    progress_print(
        "Exact duplicate sequences removed "
        "(different accession + same sequence "
        "+ same family):",
        alignment_stats[
            "duplicate_sequence_records_removed"
        ],
    )


    if alignment_stats[
        "duplicate_sequence_examples"
    ]:

        progress_print(
            "Duplicate-sequence examples "
            "(removed accession, kept accession, family):",
            alignment_stats[
                "duplicate_sequence_examples"
            ],
        )


    # ========================================================
    # FAMILY-SIZE SUMMARY
    # ========================================================

    progress_print(
        "Genomes after CSV/FASTA/sequence cleaning:",
        len(
            df_sequence_cleaned
        ),
    )


    progress_print(
        "Families after sequence cleaning:",
        df_sequence_cleaned[
            family_column
        ].nunique(),
    )


    progress_print(
        "Minimum family size required:",
        MINIMUM_FAMILY_SIZE,
    )


    progress_print(
        "Final retained genomes:",
        len(
            df_filtered
        ),
    )


    progress_print(
        "Final retained families:",
        number_of_classes,
    )


    progress_print(
        "Smallest retained family:",
        int(
            class_counts.min()
        ),
    )


    progress_print(
        "Largest retained family:",
        int(
            class_counts.max()
        ),
    )


    # --------------------------------------------------------
    # Also show how many families were excluded
    # --------------------------------------------------------

    excluded_family_counts = (
        family_counts[
            family_counts
            <
            MINIMUM_FAMILY_SIZE
        ]
    )


    progress_print(
        "Families excluded for having fewer than "
        f"{MINIMUM_FAMILY_SIZE} genomes:",
        len(
            excluded_family_counts
        ),
    )


    progress_print(
        "Genomes belonging to excluded families:",
        int(
            excluded_family_counts.sum()
        ),
    )


    # --------------------------------------------------------
    # Print every retained family and count
    # --------------------------------------------------------

    retained_family_counts = (
        df_filtered[
            family_column
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )


    progress_print(
        "\nRETAINED FAMILY COUNTS"
    )

    progress_print(
        "-" * 120
    )


    for rank, (
        family,
        count,
    ) in enumerate(
        retained_family_counts.items(),
        start=1,
    ):

        progress_print(
            f"{rank:4d} | "
            f"{family:<50} | "
            f"{int(count):6d}"
        )


    # ========================================================
    # 5. MAP FACET FILES
    # ========================================================

    facet_file_maps = {}


    for k in FACET_K_VALUES:

        directory = (
            get_facet_directory(
                k
            )
        )


        if not directory.exists():

            raise FileNotFoundError(
                f"Missing facet directory:\n"
                f"{directory}"
            )


        (
            files,
            mapping,
        ) = build_feature_mapping(

            directory=directory,

            pattern=(
                f"*_{FACET_FEATURE_NAME}.npy"
            ),

            required_set=(
                required_accession_set
            ),

            required_accessions=(
                required_accessions
            ),

            description=(
                f"PSRT facet k={k}"
            ),
        )


        facet_file_maps[
            k
        ] = mapping


        progress_print(
            f"Facet k={k}: mapped "
            f"{len(mapping)} / "
            f"{len(required_accessions)} "
            f"required files; "
            f"total files in directory="
            f"{len(files)}"
        )


    # ========================================================
    # 6. MAP K-MER FILES
    # ========================================================

    kmer_file_maps = {}


    for k in KMER_K_VALUES:

        directory = (
            get_kmer_directory(
                k
            )
        )


        if not directory.exists():

            raise FileNotFoundError(
                f"Missing k-mer directory:\n"
                f"{directory}"
            )


        (
            files,
            mapping,
        ) = build_feature_mapping(

            directory=directory,

            pattern="*.npy",

            required_set=(
                required_accession_set
            ),

            required_accessions=(
                required_accessions
            ),

            description=(
                f"ordinary k-mer k={k}"
            ),
        )


        kmer_file_maps[
            k
        ] = mapping


        progress_print(
            f"K-mer k={k}: mapped "
            f"{len(mapping)} / "
            f"{len(required_accessions)} "
            f"required files; "
            f"total files in directory="
            f"{len(files)}"
        )


    # ========================================================
    # 7. LOAD RAW FACET ARRAYS
    # ========================================================

    raw_facet_by_k = {}


    for k in FACET_K_VALUES:

        shape = expected_facet_shape(
            k
        )


        x_k = np.empty(
            (
                len(
                    required_accessions
                ),
                shape[0],
                shape[1],
            ),
            dtype=np.float32,
        )


        mapping = (
            facet_file_maps[
                k
            ]
        )


        progress_print(
            f"Loading raw facet k={k}..."
        )


        for (
            genome_index,
            accession,
        ) in enumerate(
            required_accessions
        ):

            path = mapping[
                accession
            ]


            array = np.asarray(

                np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=False,
                ),

                dtype=np.float32,
            )


            if tuple(
                array.shape
            ) != shape:

                raise ValueError(
                    f"Unexpected facet shape:\n"
                    f"k={k}\n"
                    f"{accession}\n"
                    f"Expected={shape}\n"
                    f"Received={array.shape}"
                )


            if not np.isfinite(
                array
            ).all():

                raise ValueError(
                    f"Non-finite facet values:\n"
                    f"{path}"
                )


            x_k[
                genome_index
            ] = array


        raw_facet_by_k[
            k
        ] = x_k


        progress_print(
            "  Shape:",
            x_k.shape,
        )


    # ========================================================
    # 8. LOAD AND NORMALIZE ORDINARY K-MER COUNTS
    # ========================================================

    kmer_by_k = {}


    for k in KMER_K_VALUES:

        shape = expected_kmer_shape(
            k
        )


        x_k = np.empty(
            (
                len(
                    required_accessions
                ),
                shape[0],
            ),
            dtype=np.float32,
        )


        mapping = (
            kmer_file_maps[
                k
            ]
        )


        progress_print(
            f"Loading ordinary k-mer k={k}..."
        )


        for (
            genome_index,
            accession,
        ) in enumerate(
            required_accessions
        ):

            path = mapping[
                accession
            ]


            array = np.asarray(

                np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=False,
                ),

                dtype=np.float32,
            )


            if tuple(
                array.shape
            ) != shape:

                raise ValueError(
                    f"Unexpected k-mer shape:\n"
                    f"k={k}\n"
                    f"{accession}\n"
                    f"Expected={shape}\n"
                    f"Received={array.shape}"
                )


            if not np.isfinite(
                array
            ).all():

                raise ValueError(
                    f"Non-finite k-mer values:\n"
                    f"{path}"
                )


            x_k[
                genome_index
            ] = array


        # ----------------------------------------------------
        # Normalize k-mer count vectors
        # ----------------------------------------------------

        row_sums = (
            x_k.sum(
                axis=1,
                keepdims=True,
            )
        )


        row_sums[
            row_sums == 0
        ] = 1.0


        x_k /= row_sums


        kmer_by_k[
            k
        ] = x_k


        progress_print(
            "  Shape:",
            x_k.shape,
        )


    # ========================================================
    # COMPLETED
    # ========================================================

    progress_print(
        "=" * 120
    )

    progress_print(
        "Dataset validation and feature loading "
        "completed successfully."
    )

    progress_print(
        "=" * 120
    )


    return {

        "df_filtered":
            df_filtered,

        "accession_column":
            accession_column,

        "family_column":
            family_column,

        "required_accessions":
            required_accessions,

        "y":
            y,

        "family_names":
            family_names,

        "number_of_classes":
            number_of_classes,

        "raw_facet_by_k":
            raw_facet_by_k,

        "kmer_by_k":
            kmer_by_k,
    }


# ============================================================
# FOLD-SPECIFIC FACET NORMALIZATION
# ============================================================

def normalize_facets_for_fold(
    raw_facet_by_k,
    train_indices,
):
    """
    Min-max normalization is fitted only on the training
    subset of the current fold.

    The same training-derived minimum and maximum are then
    used for validation, reference, and held-out test samples.
    """

    normalized = {}


    for k in FACET_K_VALUES:

        raw = (
            raw_facet_by_k[
                k
            ]
        )


        # ----------------------------------------------------
        # Training-only min/max
        # ----------------------------------------------------

        training_minimum = (
            raw[
                train_indices
            ]
            .min(
                axis=0
            )
        )


        training_maximum = (
            raw[
                train_indices
            ]
            .max(
                axis=0
            )
        )


        training_range = (
            training_maximum
            -
            training_minimum
        )


        training_range[
            training_range == 0
        ] = 1.0


        # ----------------------------------------------------
        # Apply training-derived normalization to all samples
        # ----------------------------------------------------

        x_k = np.empty_like(
            raw,
            dtype=np.float32,
        )


        np.subtract(
            raw,
            training_minimum[
                None,
                :,
                :,
            ],
            out=x_k,
        )


        np.divide(
            x_k,
            training_range[
                None,
                :,
                :,
            ],
            out=x_k,
        )


        np.clip(
            x_k,
            0.0,
            1.0,
            out=x_k,
        )


        normalized[
            k
        ] = x_k


    return normalized
