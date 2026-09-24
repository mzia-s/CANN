#!/usr/bin/env python3

import os
import re
import argparse
from collections import Counter

import numpy as np
from Bio import SeqIO

from psrt import *
from kmer_only_encoder import kmer_counts_encoder


# =============================================================================
# FIXED FEATURE CONFIGURATION
# =============================================================================

ALPHABET = ["A", "C", "G", "T"]

# Final selected feature set
PSRT_K_VALUES = (3, 4, 5)
KMER_COUNT_K_VALUES = (6, 7)

# Only dimension 0 is needed.
MAX_DIMENSION = 0

# Exactly two filtration positions:
# k=3 -> [0, 64]
# k=4 -> [0, 256]
# k=5 -> [0, 1024]
NUMBER_OF_FILTRATION_POSITIONS = 2

VALID_DNA_CHARACTERS = frozenset("ACGT")


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate final CAKR features, count usable genomes, "
            "or validate all generated features."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["worker", "count", "validate"],
        help=(
            "worker = process one cleaned unique genome; "
            "count = report number of cleaned unique genomes; "
            "validate = verify all generated features"
        ),
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file",
    )

    parser.add_argument(
        "--out_root",
        required=True,
        help="Root directory for generated features",
    )

    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Cleaned unique genome index for worker mode",
    )

    return parser.parse_args()


# =============================================================================
# ACCESSION EXTRACTION
# =============================================================================

def extract_accession(text):
    """
    Try to find an accession such as:

    PZ251736.1
    NC_007373.1
    GCA_052313375.1
    """
    match = re.search(
        r"([A-Z]{1,5}_?\d+\.\d+)",
        str(text),
    )

    if match:
        return match.group(1)

    return None


def get_record_accession(record):
    """
    Return the canonical accession used for feature filenames.

    For NCBI/Yau datasets this should be the accession with version,
    e.g. NC_031030.2. If no recognizable accession exists, fall back
    to record.id so the record is still addressable.
    """
    accession = (
        extract_accession(record.id)
        or extract_accession(record.description)
    )

    if accession is not None:
        return accession

    return record.id.strip()


# =============================================================================
# SEQUENCE CLEANING
# =============================================================================

def clean_dna_sequence(raw_sequence):
    """
    Clean one DNA/RNA sequence using the rule required for this experiment.

    1. Convert to uppercase.
    2. Convert U -> T.
    3. REMOVE every remaining character that is not A, C, G, or T.
       Example: ACNT -> ACT.
    4. Return the cleaned sequence and cleaning statistics.

    The entire genome is NOT discarded just because it contains N, R, Y,
    gaps, or another nonstandard symbol. A genome is unusable only if the
    cleaned sequence becomes empty.
    """
    sequence = str(raw_sequence).upper()

    u_count = sequence.count("U")
    if u_count:
        sequence = sequence.replace("U", "T")

    removed_characters = Counter(
        character
        for character in sequence
        if character not in VALID_DNA_CHARACTERS
    )

    cleaned_sequence = "".join(
        character
        for character in sequence
        if character in VALID_DNA_CHARACTERS
    )

    return (
        cleaned_sequence,
        u_count,
        removed_characters,
    )


# =============================================================================
# FASTA LOADING + SAFE DEDUPLICATION
# =============================================================================

def load_sequences(
    fasta_path,
    print_summary=False,
):
    """
    Read, clean, and deduplicate the FASTA.

    Rules
    -----
    * U -> T.
    * Non-ACGT characters are removed individually.
    * Empty-after-cleaning records are skipped.
    * Same accession + same cleaned sequence -> keep one.
    * Same accession + different cleaned sequence -> ERROR.
    * Different accessions are BOTH retained even if their cleaned sequences
      are identical. Family-aware exact-sequence deduplication is performed
      later by data.py, because feature generation does not know the family.

    The final accession list is unique.
    """
    records = list(
        SeqIO.parse(
            fasta_path,
            "fasta",
        )
    )

    if len(records) == 0:
        raise ValueError(
            f"No sequences found in FASTA: {fasta_path}"
        )

    raw_record_count = len(records)

    sequence_by_accession = {}
    accession_order = []

    sequence_to_first_accession = {}

    duplicate_same_accession_same_sequence = []
    duplicate_different_accession_same_sequence_detected = []
    conflicting_same_accession_sequences = []

    empty_after_cleaning = []

    sequences_with_u = 0
    total_u_converted = 0

    sequences_with_removed_characters = 0
    total_removed_characters = 0
    removed_character_counts = Counter()
    cleaned_examples = []

    for record in records:
        accession = get_record_accession(record)

        (
            cleaned_sequence,
            u_count,
            removed_characters,
        ) = clean_dna_sequence(
            record.seq
        )

        if u_count > 0:
            sequences_with_u += 1
            total_u_converted += u_count

        if removed_characters:
            sequences_with_removed_characters += 1
            removed_count = sum(
                removed_characters.values()
            )
            total_removed_characters += removed_count
            removed_character_counts.update(
                removed_characters
            )

            if len(cleaned_examples) < 20:
                cleaned_examples.append(
                    (
                        accession,
                        "".join(
                            sorted(
                                removed_characters.keys()
                            )
                        ),
                        removed_count,
                        len(str(record.seq)),
                        len(cleaned_sequence),
                    )
                )

        if len(cleaned_sequence) == 0:
            empty_after_cleaning.append(
                accession
            )
            continue

        # ---------------------------------------------------------------------
        # Same accession seen before
        # ---------------------------------------------------------------------
        if accession in sequence_by_accession:
            previous_sequence = sequence_by_accession[
                accession
            ]

            if previous_sequence == cleaned_sequence:
                duplicate_same_accession_same_sequence.append(
                    accession
                )
                continue

            conflicting_same_accession_sequences.append(
                accession
            )
            continue

        # ---------------------------------------------------------------------
        # Different accession with exact same cleaned sequence
        # ---------------------------------------------------------------------
        # Keep BOTH accessions here. Feature generation does not know the
        # family label, so it cannot safely decide whether this is a removable
        # same-family duplicate or a conflicting cross-family duplicate.
        if cleaned_sequence in sequence_to_first_accession:
            duplicate_different_accession_same_sequence_detected.append(
                (
                    accession,
                    sequence_to_first_accession[cleaned_sequence],
                )
            )
        else:
            sequence_to_first_accession[cleaned_sequence] = accession

        # ---------------------------------------------------------------------
        # New unique accession
        # ---------------------------------------------------------------------
        sequence_by_accession[
            accession
        ] = cleaned_sequence

        accession_order.append(
            accession
        )

    if conflicting_same_accession_sequences:
        conflicts = list(
            dict.fromkeys(
                conflicting_same_accession_sequences
            )
        )

        raise ValueError(
            "The same accession occurs in the FASTA with different "
            "cleaned nucleotide sequences.\n"
            "The pipeline will not choose one arbitrarily.\n"
            f"Examples: {conflicts[:20]}"
        )

    accessions = accession_order

    sequences = [
        sequence_by_accession[
            accession
        ]
        for accession in accessions
    ]

    if len(accessions) != len(sequences):
        raise RuntimeError(
            "Internal FASTA preprocessing error: accession and "
            "sequence counts differ."
        )

    if len(accessions) != len(set(accessions)):
        raise RuntimeError(
            "Internal FASTA preprocessing error: duplicate accessions remain."
        )

    stats = {
        "raw_fasta_records":
            raw_record_count,

        "usable_unique_genomes":
            len(accessions),

        "same_accession_same_sequence_duplicates_removed":
            len(
                duplicate_same_accession_same_sequence
            ),

        "same_accession_same_sequence_examples":
            duplicate_same_accession_same_sequence[
                :20
            ],

        "different_accession_same_sequence_duplicates_detected":
            len(
                duplicate_different_accession_same_sequence_detected
            ),

        "different_accession_same_sequence_examples":
            duplicate_different_accession_same_sequence_detected[
                :20
            ],

        "empty_after_cleaning_skipped":
            len(
                empty_after_cleaning
            ),

        "empty_after_cleaning_examples":
            empty_after_cleaning[
                :20
            ],

        "sequences_with_u":
            sequences_with_u,

        "total_u_converted":
            total_u_converted,

        "sequences_cleaned_nonstandard":
            sequences_with_removed_characters,

        "total_nonstandard_characters_removed":
            total_removed_characters,

        "removed_character_counts":
            dict(
                sorted(
                    removed_character_counts.items()
                )
            ),

        "cleaned_examples":
            cleaned_examples,
    }

    if print_summary:
        print_fasta_preprocessing_summary(
            stats
        )

    return (
        accessions,
        sequences,
        stats,
    )


def print_fasta_preprocessing_summary(
    stats,
):
    print("=" * 100)
    print("FASTA PREPROCESSING")
    print("=" * 100)

    print(
        f"Raw FASTA records                         : "
        f"{stats['raw_fasta_records']}"
    )

    print(
        f"Usable unique genomes                     : "
        f"{stats['usable_unique_genomes']}"
    )

    print(
        f"Same-accession duplicate records removed  : "
        f"{stats['same_accession_same_sequence_duplicates_removed']}"
    )

    print(
        f"Different-accession duplicate seqs detected: "
        f"{stats['different_accession_same_sequence_duplicates_detected']}"
    )

    print(
        f"Sequences cleaned for nonstandard letters : "
        f"{stats['sequences_cleaned_nonstandard']}"
    )

    print(
        f"Nonstandard letters removed               : "
        f"{stats['total_nonstandard_characters_removed']}"
    )

    print(
        f"Empty sequences skipped after cleaning    : "
        f"{stats['empty_after_cleaning_skipped']}"
    )

    print(
        f"Sequences containing U converted to T     : "
        f"{stats['sequences_with_u']}"
    )

    print(
        f"Total U letters converted to T            : "
        f"{stats['total_u_converted']}"
    )

    if stats["removed_character_counts"]:
        print()
        print("Removed nonstandard characters:")

        for character, count in (
            stats[
                "removed_character_counts"
            ].items()
        ):
            print(
                f"  {character!r} : "
                f"{count} occurrence(s)"
            )

    if stats["cleaned_examples"]:
        print()
        print(
            "Examples of cleaned genomes "
            "(accession | removed | count | old_len -> new_len):"
        )

        for (
            accession,
            characters,
            count,
            old_length,
            new_length,
        ) in stats[
            "cleaned_examples"
        ]:
            print(
                f"  {accession} | "
                f"{characters!r} | "
                f"{count} | "
                f"{old_length} -> {new_length}"
            )

    if stats[
        "same_accession_same_sequence_examples"
    ]:
        print()
        print(
            "Same-accession exact duplicate "
            "records removed:"
        )

        for accession in stats[
            "same_accession_same_sequence_examples"
        ]:
            print(
                f"  {accession}"
            )

    if stats[
        "different_accession_same_sequence_examples"
    ]:
        print()
        print(
            "Different-accession exact sequence "
            "duplicates detected and retained "
            "(accession -> first matching accession):"
        )

        for (
            removed_accession,
            kept_accession,
        ) in stats[
            "different_accession_same_sequence_examples"
        ]:
            print(
                f"  {removed_accession} -> "
                f"{kept_accession}"
            )

    if stats[
        "empty_after_cleaning_examples"
    ]:
        print()
        print(
            "Empty-after-cleaning genomes skipped:"
        )

        for accession in stats[
            "empty_after_cleaning_examples"
        ]:
            print(
                f"  {accession}"
            )

    print("=" * 100)


# =============================================================================
# SAFE OVERWRITE
# =============================================================================

def save_npy_overwrite(
    path,
    array,
):
    directory = os.path.dirname(
        path
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    if os.path.exists(
        path
    ):
        os.remove(
            path
        )

    np.save(
        path,
        array,
    )


# =============================================================================
# PSRT FACET0
# =============================================================================

def calculate_psrt_facet0(
    dna,
    k,
):
    """
    Calculate facet0 for one genome and one k.

    Filtration:
        [0, 4^k]

    Shapes:
        k=3 -> (64, 2)
        k=4 -> (256, 2)
        k=5 -> (1024, 2)
    """
    number_of_kmers = (
        4 ** k
    )

    specific_filtration = np.array(
        [
            0.0,
            float(
                number_of_kmers
            ),
        ],
        dtype=float,
    )

    if (
        specific_filtration.shape
        !=
        (
            NUMBER_OF_FILTRATION_POSITIONS,
        )
    ):
        raise ValueError(
            "Unexpected filtration shape: "
            f"{specific_filtration.shape}"
        )

    kmers = generate_all_kmers(
        ALPHABET,
        k,
    )

    if (
        len(kmers)
        !=
        number_of_kmers
    ):
        raise ValueError(
            f"k={k}: expected "
            f"{number_of_kmers} k-mers, "
            f"but generate_all_kmers "
            f"returned {len(kmers)}."
        )

    facet_matrix = np.zeros(
        (
            number_of_kmers,
            NUMBER_OF_FILTRATION_POSITIONS,
        ),
        dtype=float,
    )

    for (
        kmer_index,
        kmer,
    ) in enumerate(
        kmers
    ):
        points = occurrence(
            dna,
            kmer,
        )

        if len(
            points
        ) == 0:
            continue

        points = np.asarray(
            points,
            dtype=float,
        )[
            :,
            np.newaxis,
        ]

        ph = PH(
            points,
            max_dimension=(
                MAX_DIMENSION
            ),
            max_edge_length=2.0,
            specific_filtration=(
                specific_filtration
            ),
        )

        alphas, _ = (
            ph.betti_curves()
        )

        alphas = np.asarray(
            alphas,
            dtype=float,
        ).ravel()

        if (
            alphas.shape
            !=
            (
                NUMBER_OF_FILTRATION_POSITIONS,
            )
        ):
            raise ValueError(
                f"k={k}, k-mer={kmer}: "
                f"expected 2 filtration "
                f"positions, received "
                f"{alphas.shape}."
            )

        facet_curves = (
            ph.facet_curves()
        )

        facet_curve = (
            facet_curves.get(
                0,
                None,
            )
        )

        if facet_curve is None:
            continue

        facet_curve = np.asarray(
            facet_curve,
            dtype=float,
        ).ravel()

        if (
            facet_curve.shape
            !=
            (
                NUMBER_OF_FILTRATION_POSITIONS,
            )
        ):
            raise ValueError(
                f"k={k}, k-mer={kmer}: "
                f"expected facet0 shape "
                f"(2,), received "
                f"{facet_curve.shape}."
            )

        facet_matrix[
            kmer_index,
            :,
        ] = facet_curve

    expected_shape = (
        number_of_kmers,
        NUMBER_OF_FILTRATION_POSITIONS,
    )

    if (
        facet_matrix.shape
        !=
        expected_shape
    ):
        raise ValueError(
            f"k={k}: expected final "
            f"facet shape "
            f"{expected_shape}, "
            f"received "
            f"{facet_matrix.shape}."
        )

    return facet_matrix


# =============================================================================
# ORDINARY K-MER COUNTS
# =============================================================================

def calculate_kmer_counts(
    dna,
    k,
):
    model = kmer_counts_encoder(
        k=k
    )

    features = np.asarray(
        list(
            model.analyze(
                dna
            ).values()
        ),
        dtype=float,
    )

    expected_shape = (
        4 ** k,
    )

    if (
        features.shape
        !=
        expected_shape
    ):
        raise ValueError(
            f"k-mer count k={k}: "
            f"expected shape "
            f"{expected_shape}, "
            f"received "
            f"{features.shape}."
        )

    return features


# =============================================================================
# PROCESS ONE CLEANED UNIQUE GENOME
# =============================================================================

def process_one_genome(
    accession,
    dna,
    sample_index,
    out_root,
):
    print("=" * 80)

    print(
        f"Genome index   : "
        f"{sample_index}"
    )

    print(
        f"Genome         : "
        f"{accession}"
    )

    print(
        f"Sequence length: "
        f"{len(dna)}"
    )

    # =========================================================================
    # PSRT facet0 k=3,4,5
    # =========================================================================

    for k in (
        PSRT_K_VALUES
    ):
        print("-" * 80)

        print(
            f"PSRT facet0 k={k}"
        )

        print(
            f"Filtration = "
            f"[0, {4 ** k}]"
        )

        facet_matrix = (
            calculate_psrt_facet0(
                dna=dna,
                k=k,
            )
        )

        save_directory = os.path.join(
            out_root,
            "output_psrt",
            "all_features",
            f"k{k}",
        )

        save_path = os.path.join(
            save_directory,
            f"{accession}_facet0.npy",
        )

        save_npy_overwrite(
            save_path,
            facet_matrix,
        )

        print(
            f"Saved shape: "
            f"{facet_matrix.shape}"
        )

    # =========================================================================
    # k-mer counts k=6,7
    # =========================================================================

    for k in (
        KMER_COUNT_K_VALUES
    ):
        print("-" * 80)

        print(
            f"k-mer counts k={k}"
        )

        count_features = (
            calculate_kmer_counts(
                dna=dna,
                k=k,
            )
        )

        save_directory = os.path.join(
            out_root,
            "output_kmer",
            "kmer_counts",
            str(k),
        )

        save_path = os.path.join(
            save_directory,
            f"{accession}.npy",
        )

        save_npy_overwrite(
            save_path,
            count_features,
        )

        print(
            f"Saved shape: "
            f"{count_features.shape}"
        )

    print("-" * 80)

    print(
        f"COMPLETE: "
        f"{accession}"
    )


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def check_npy_file(
    path,
    expected_shape,
):
    if not os.path.exists(
        path
    ):
        return (
            False,
            "missing",
        )

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

    except Exception as error:
        return (
            False,
            f"cannot load: {error}",
        )

    if (
        array.shape
        !=
        expected_shape
    ):
        return (
            False,
            (
                f"wrong shape "
                f"{array.shape}; "
                f"expected "
                f"{expected_shape}"
            ),
        )

    if not np.isfinite(
        array
    ).all():
        return (
            False,
            "contains NaN or infinite values",
        )

    return (
        True,
        "ok",
    )


# =============================================================================
# VALIDATE ENTIRE CLEANED DATASET
# =============================================================================

def validate_dataset(
    fasta_path,
    out_root,
):
    (
        accessions,
        sequences,
        _,
    ) = load_sequences(
        fasta_path,
        print_summary=True,
    )

    total_genomes = len(
        sequences
    )

    print()
    print("=" * 100)
    print(
        "FINAL FEATURE VALIDATION"
    )
    print("=" * 100)

    print(
        f"Usable FASTA genomes: "
        f"{total_genomes}"
    )

    print()

    feature_counts = {
        "PSRT k=3": 0,
        "PSRT k=4": 0,
        "PSRT k=5": 0,
        "kmer k=6": 0,
        "kmer k=7": 0,
    }

    complete_genomes = 0
    problems = []

    for accession in accessions:
        genome_complete = True

        # ---------------------------------------------------------------------
        # PSRT facet0
        # ---------------------------------------------------------------------
        for k in (
            PSRT_K_VALUES
        ):
            path = os.path.join(
                out_root,
                "output_psrt",
                "all_features",
                f"k{k}",
                f"{accession}_facet0.npy",
            )

            expected_shape = (
                4 ** k,
                NUMBER_OF_FILTRATION_POSITIONS,
            )

            valid, reason = (
                check_npy_file(
                    path,
                    expected_shape,
                )
            )

            key = (
                f"PSRT k={k}"
            )

            if valid:
                feature_counts[
                    key
                ] += 1

            else:
                genome_complete = (
                    False
                )

                problems.append(
                    (
                        accession,
                        key,
                        reason,
                    )
                )

        # ---------------------------------------------------------------------
        # k-mer counts
        # ---------------------------------------------------------------------
        for k in (
            KMER_COUNT_K_VALUES
        ):
            path = os.path.join(
                out_root,
                "output_kmer",
                "kmer_counts",
                str(k),
                f"{accession}.npy",
            )

            expected_shape = (
                4 ** k,
            )

            valid, reason = (
                check_npy_file(
                    path,
                    expected_shape,
                )
            )

            key = (
                f"kmer k={k}"
            )

            if valid:
                feature_counts[
                    key
                ] += 1

            else:
                genome_complete = (
                    False
                )

                problems.append(
                    (
                        accession,
                        key,
                        reason,
                    )
                )

        if genome_complete:
            complete_genomes += 1

    print(
        f"PSRT k=3          : "
        f"{feature_counts['PSRT k=3']}/"
        f"{total_genomes}"
    )

    print(
        f"PSRT k=4          : "
        f"{feature_counts['PSRT k=4']}/"
        f"{total_genomes}"
    )

    print(
        f"PSRT k=5          : "
        f"{feature_counts['PSRT k=5']}/"
        f"{total_genomes}"
    )

    print(
        f"k-mer counts k=6  : "
        f"{feature_counts['kmer k=6']}/"
        f"{total_genomes}"
    )

    print(
        f"k-mer counts k=7  : "
        f"{feature_counts['kmer k=7']}/"
        f"{total_genomes}"
    )

    print()

    print(
        f"Complete genomes  : "
        f"{complete_genomes}/"
        f"{total_genomes}"
    )

    print(
        f"Problems detected : "
        f"{len(problems)}"
    )

    if problems:
        print()
        print("=" * 100)
        print("PROBLEMS")
        print("=" * 100)

        maximum_to_print = (
            100
        )

        for (
            accession,
            feature,
            reason,
        ) in problems[
            :maximum_to_print
        ]:
            print(
                f"{accession} | "
                f"{feature} | "
                f"{reason}"
            )

        if (
            len(problems)
            >
            maximum_to_print
        ):
            remaining = (
                len(problems)
                -
                maximum_to_print
            )

            print(
                f"... {remaining} "
                f"additional problems "
                f"not printed."
            )

        print()
        print("=" * 100)
        print(
            "VALIDATION FAILED"
        )

        print(
            f"Only {complete_genomes} "
            f"of {total_genomes} genomes "
            f"have all required features."
        )

        print("=" * 100)

        raise SystemExit(
            1
        )

    print()
    print("=" * 100)
    print(
        "VALIDATION PASSED"
    )

    print(
        f"All {total_genomes} "
        f"usable genomes have all "
        f"required features."
    )

    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if not os.path.exists(
        args.fasta
    ):
        raise FileNotFoundError(
            f"FASTA file not found: "
            f"{args.fasta}"
        )

    # =========================================================================
    # COUNT MODE
    # =========================================================================
    if args.mode == "count":
        (
            accessions,
            _,
            _,
        ) = load_sequences(
            args.fasta,
            print_summary=True,
        )

        # Machine-readable line used by run_cakr.sh.
        print(
            f"USABLE_GENOMES="
            f"{len(accessions)}"
        )

        return

    # =========================================================================
    # WORKER MODE
    # =========================================================================
    if args.mode == "worker":
        if args.task_id is None:
            raise ValueError(
                "--task_id is required "
                "for worker mode."
            )

        (
            accessions,
            sequences,
            _,
        ) = load_sequences(
            args.fasta,
            print_summary=False,
        )

        total_genomes = len(
            sequences
        )

        if (
            args.task_id
            <
            0
            or
            args.task_id
            >=
            total_genomes
        ):
            raise IndexError(
                f"task_id={args.task_id} "
                f"is outside the valid "
                f"range 0 to "
                f"{total_genomes - 1}."
            )

        accession = accessions[
            args.task_id
        ]

        dna = sequences[
            args.task_id
        ]

        process_one_genome(
            accession=accession,
            dna=dna,
            sample_index=(
                args.task_id
            ),
            out_root=(
                args.out_root
            ),
        )

        return

    # =========================================================================
    # VALIDATION MODE
    # =========================================================================
    if args.mode == "validate":
        validate_dataset(
            fasta_path=(
                args.fasta
            ),
            out_root=(
                args.out_root
            ),
        )

        return

    raise ValueError(
        f"Unknown mode: "
        f"{args.mode}"
    )


if __name__ == "__main__":
    main()