#!/usr/bin/env python3

import argparse
import contextlib
import gc
import traceback

import numpy as np
import torch

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from config import (
    RESULT_DIR,
    CV_SEEDS,
    NUMBER_OF_FOLDS,
    VALIDATION_FRACTION_OF_DEVELOPMENT,
    INPUT_SPECS,
    CNN_MAXIMUM_EPOCHS,
    CNN_LEARNING_RATE,
    CNN_WEIGHT_DECAY,
    TRANSFORMER_MAXIMUM_EPOCHS,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_WEIGHT_DECAY,
    METHODS,
)

from data import (
    load_complete_dataset,
    normalize_facets_for_fold,
)

from dataset import (
    build_fold_loaders,
)

from cnn import (
    HybridFacetKmerCNN,
)

from transformer import (
    HybridFacetKmerTransformer,
)

from training import (
    progress_print,
    set_global_seed,
    make_run_seed,
    train_model,
    calculate_metrics,
)

from evaluation import (
    evaluate_fold_models,
)

from results import (
    metrics_to_text,
    print_fold_result_line,
    print_oof_result_line,
    aggregate_results,
)


# ============================================================
# RUN ONE SEED
#
# ONE SEED JOB = FIVE OUTER FOLDS
# ============================================================

def run_seed(
    seed_index,
):

    if (
        seed_index < 0
        or
        seed_index
        >=
        len(
            CV_SEEDS
        )
    ):

        raise ValueError(
            "seed-index must be from "
            f"0 to {len(CV_SEEDS) - 1}."
        )

    cv_seed = (
        CV_SEEDS[
            seed_index
        ]
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_DIR
        /
        f"seed_{cv_seed:02d}.txt"
    )

    # Everything produced by this seed job goes into
    # exactly one text file.

    with output_path.open(
        "w",
        buffering=1,
    ) as output_handle:

        with (
            contextlib.redirect_stdout(
                output_handle
            ),
            contextlib.redirect_stderr(
                output_handle
            ),
        ):

            try:

                progress_print(
                    "=" * 140
                )

                progress_print(
                    "REPEATED 5-FOLD CV SEED JOB"
                )

                progress_print(
                    "=" * 140
                )

                progress_print(
                    "Seed index:",
                    seed_index,
                )

                progress_print(
                    "CV seed:",
                    cv_seed,
                )

                progress_print(
                    "Number of folds:",
                    NUMBER_OF_FOLDS,
                )

                progress_print(
                    "Results file:",
                    output_path,
                )

                progress_print(
                    "Feature set: "
                    "facet k=3,4,5 + "
                    "k-mer k=6,7"
                )

                progress_print(
                    "Consensus: "
                    "CNN=0.50, "
                    "Transformer=0.50"
                )

                progress_print(
                    "Checkpoint: validation "
                    "encoder cosine 1-NN ACC; "
                    "F1 tie-break"
                )

                progress_print(
                    "Final reference in each fold: "
                    "train + validation"
                )


                # ============================================
                # DEVICE
                # ============================================

                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                progress_print(
                    "Device:",
                    device,
                )


                # ============================================
                # LOAD COMPLETE DATASET
                # ============================================

                data = (
                    load_complete_dataset()
                )

                y = (
                    data[
                        "y"
                    ]
                )

                raw_facet_by_k = (
                    data[
                        "raw_facet_by_k"
                    ]
                )

                kmer_by_k = (
                    data[
                        "kmer_by_k"
                    ]
                )

                number_of_classes = (
                    data[
                        "number_of_classes"
                    ]
                )

                all_indices = np.arange(
                    len(y),
                    dtype=np.int64,
                )


                # ============================================
                # OUTER 5-FOLD CV
                # ============================================

                outer_cv = StratifiedKFold(
                    n_splits=(
                        NUMBER_OF_FOLDS
                    ),
                    shuffle=True,
                    random_state=(
                        cv_seed
                    ),
                )


                # ============================================
                # OOF STORAGE
                # ============================================

                oof_predictions = {

                    method:
                        np.full(
                            len(y),
                            -1,
                            dtype=np.int64,
                        )

                    for method
                    in METHODS
                }

                oof_assignment_count = np.zeros(
                    len(y),
                    dtype=np.int64,
                )

                fold_records = []


                # ============================================
                # FIVE FOLDS
                # ============================================

                for fold_number, (
                    development_indices,
                    test_indices,
                ) in enumerate(

                    outer_cv.split(
                        all_indices,
                        y,
                    ),

                    start=1,
                ):

                    development_indices = np.asarray(
                        development_indices,
                        dtype=np.int64,
                    )

                    test_indices = np.asarray(
                        test_indices,
                        dtype=np.int64,
                    )

                    fold_seed = make_run_seed(
                        cv_seed,
                        fold_number,
                    )


                    # ========================================
                    # INNER 70 / 10 SPLIT
                    # ========================================

                    (
                        train_indices,
                        validation_indices,
                    ) = train_test_split(

                        development_indices,

                        test_size=(
                            VALIDATION_FRACTION_OF_DEVELOPMENT
                        ),

                        random_state=(
                            fold_seed
                        ),

                        shuffle=True,

                        stratify=(
                            y[
                                development_indices
                            ]
                        ),
                    )

                    train_indices = np.asarray(
                        train_indices,
                        dtype=np.int64,
                    )

                    validation_indices = np.asarray(
                        validation_indices,
                        dtype=np.int64,
                    )

                    reference_indices = np.concatenate(
                        [
                            train_indices,
                            validation_indices,
                        ]
                    ).astype(
                        np.int64,
                        copy=False,
                    )


                    # ========================================
                    # SPLIT SAFETY CHECKS
                    # ========================================

                    if (
                        np.intersect1d(
                            train_indices,
                            validation_indices,
                        ).size
                        !=
                        0
                    ):

                        raise RuntimeError(
                            "Train/validation "
                            "overlap detected."
                        )

                    if (
                        np.intersect1d(
                            train_indices,
                            test_indices,
                        ).size
                        !=
                        0
                    ):

                        raise RuntimeError(
                            "Train/test overlap detected."
                        )

                    if (
                        np.intersect1d(
                            validation_indices,
                            test_indices,
                        ).size
                        !=
                        0
                    ):

                        raise RuntimeError(
                            "Validation/test overlap detected."
                        )

                    if (
                        len(
                            reference_indices
                        )
                        +
                        len(
                            test_indices
                        )
                        !=
                        len(y)
                    ):

                        raise RuntimeError(
                            "Reference + test does not "
                            "cover complete dataset."
                        )


                    # ========================================
                    # FOLD INFORMATION
                    # ========================================

                    progress_print(
                        "\n"
                        +
                        "=" * 140
                    )

                    progress_print(
                        f"SEED {cv_seed} | "
                        f"FOLD {fold_number}/"
                        f"{NUMBER_OF_FOLDS}"
                    )

                    progress_print(
                        "=" * 140
                    )

                    progress_print(
                        "Run seed:",
                        fold_seed,
                    )

                    progress_print(
                        "Train:",
                        len(
                            train_indices
                        ),
                    )

                    progress_print(
                        "Validation:",
                        len(
                            validation_indices
                        ),
                    )

                    progress_print(
                        "Reference train+validation:",
                        len(
                            reference_indices
                        ),
                    )

                    progress_print(
                        "Held-out test fold:",
                        len(
                            test_indices
                        ),
                    )

                    progress_print(
                        "Approximate percentages: "
                        f"train="
                        f"{100 * len(train_indices) / len(y):.2f}% "
                        f"validation="
                        f"{100 * len(validation_indices) / len(y):.2f}% "
                        f"test="
                        f"{100 * len(test_indices) / len(y):.2f}%"
                    )


                    # ========================================
                    # TRAINING-ONLY FACET NORMALIZATION
                    # ========================================

                    facet_by_k = (
                        normalize_facets_for_fold(
                            raw_facet_by_k,
                            train_indices,
                        )
                    )


                    # ========================================
                    # DATA LOADERS
                    # ========================================

                    loaders = build_fold_loaders(

                        facet_data=(
                            facet_by_k
                        ),

                        kmer_data=(
                            kmer_by_k
                        ),

                        y=y,

                        train_indices=(
                            train_indices
                        ),

                        validation_indices=(
                            validation_indices
                        ),

                        reference_indices=(
                            reference_indices
                        ),

                        test_indices=(
                            test_indices
                        ),

                        sampler_seed=(
                            fold_seed
                        ),
                    )


                    # ========================================
                    # CNN
                    # ========================================

                    set_global_seed(
                        fold_seed
                    )

                    cnn_model = HybridFacetKmerCNN(
                        INPUT_SPECS,
                        number_of_classes,
                    ).to(
                        device
                    )

                    (
                        cnn_model,
                        best_cnn_epoch,
                        _,
                    ) = train_model(

                        cnn_model,

                        "CNN",

                        CNN_MAXIMUM_EPOCHS,

                        CNN_LEARNING_RATE,

                        CNN_WEIGHT_DECAY,

                        loaders,

                        device,
                    )


                    # ========================================
                    # CLEAN CACHE
                    # ========================================

                    gc.collect()

                    if torch.cuda.is_available():

                        torch.cuda.empty_cache()


                    # ========================================
                    # TRANSFORMER
                    # ========================================

                    set_global_seed(
                        fold_seed
                    )

                    transformer_model = (
                        HybridFacetKmerTransformer(
                            INPUT_SPECS,
                            number_of_classes,
                        )
                        .to(
                            device
                        )
                    )

                    (
                        transformer_model,
                        best_transformer_epoch,
                        _,
                    ) = train_model(

                        transformer_model,

                        "Transformer",

                        TRANSFORMER_MAXIMUM_EPOCHS,

                        TRANSFORMER_LEARNING_RATE,

                        TRANSFORMER_WEIGHT_DECAY,

                        loaders,

                        device,
                    )


                    # ========================================
                    # HELD-OUT TEST FOLD
                    # ========================================

                    (
                        predictions,
                        fold_metrics,
                        test_labels,
                    ) = evaluate_fold_models(

                        cnn_model,

                        transformer_model,

                        loaders,

                        number_of_classes,

                        device,
                    )


                    # ========================================
                    # TEST ORDER CHECK
                    # ========================================

                    if not np.array_equal(
                        test_labels,
                        y[
                            test_indices
                        ],
                    ):

                        raise RuntimeError(
                            "Test labels returned by "
                            "loader do not match "
                            "y[test_indices]."
                        )


                    # ========================================
                    # STORE OOF PREDICTIONS
                    # ========================================

                    for method in METHODS:

                        oof_predictions[
                            method
                        ][
                            test_indices
                        ] = (
                            predictions[
                                method
                            ]
                        )

                    oof_assignment_count[
                        test_indices
                    ] += 1


                    # ========================================
                    # PRINT FOLD RESULTS
                    # ========================================

                    progress_print(
                        "\nFOLD TEST RESULTS"
                    )

                    for method in METHODS:

                        progress_print(
                            f"{method:18s} | "
                            +
                            metrics_to_text(
                                fold_metrics[
                                    method
                                ]
                            )
                        )

                        print_fold_result_line(
                            cv_seed,
                            fold_number,
                            method,
                            fold_metrics[
                                method
                            ],
                            best_cnn_epoch,
                            best_transformer_epoch,
                        )

                    fold_records.append(
                        {

                            "fold":
                                fold_number,

                            "cnn_epoch":
                                best_cnn_epoch,

                            "transformer_epoch":
                                best_transformer_epoch,

                            "test_size":
                                len(
                                    test_indices
                                ),
                        }
                    )


                    # ========================================
                    # CLEAN FOLD OBJECTS
                    # ========================================

                    del cnn_model

                    del transformer_model

                    del loaders

                    del facet_by_k

                    del predictions

                    gc.collect()

                    if torch.cuda.is_available():

                        torch.cuda.empty_cache()


                # ============================================
                # VERIFY COMPLETE OOF COVERAGE
                # ============================================

                if not np.all(
                    oof_assignment_count
                    ==
                    1
                ):

                    bad = np.where(
                        oof_assignment_count
                        !=
                        1
                    )[0][
                        :20
                    ]

                    raise RuntimeError(
                        "OOF assignment error. "
                        "Every genome must appear in "
                        "exactly one held-out fold. "
                        f"Bad indices: {bad.tolist()}"
                    )


                # ============================================
                # COMPLETE OOF RESULTS FOR THIS SEED
                # ============================================

                progress_print(
                    "\n"
                    +
                    "=" * 140
                )

                progress_print(
                    f"SEED {cv_seed} "
                    "COMPLETE OOF RESULTS"
                )

                progress_print(
                    "=" * 140
                )

                for method in METHODS:

                    if np.any(
                        oof_predictions[
                            method
                        ]
                        <
                        0
                    ):

                        raise RuntimeError(
                            "Missing OOF predictions "
                            f"for {method}."
                        )

                    metrics = calculate_metrics(
                        y,
                        oof_predictions[
                            method
                        ],
                    )

                    progress_print(
                        f"{method:18s} | "
                        +
                        metrics_to_text(
                            metrics
                        )
                    )

                    print_oof_result_line(
                        cv_seed,
                        method,
                        metrics,
                    )


                # ============================================
                # SELECTED EPOCHS
                # ============================================

                progress_print(
                    "\nSELECTED EPOCHS BY FOLD"
                )

                for record in fold_records:

                    progress_print(
                        f"Fold "
                        f"{record['fold']}: "
                        f"CNN="
                        f"{record['cnn_epoch']} | "
                        f"Transformer="
                        f"{record['transformer_epoch']} | "
                        f"test_size="
                        f"{record['test_size']}"
                    )

                progress_print(
                    "\nSEED JOB COMPLETED SUCCESSFULLY"
                )

            except Exception:

                progress_print(
                    "\nSEED JOB FAILED"
                )

                traceback.print_exc()

                raise


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--seed-index",
        type=int,
        help=(
            "Array-task index from 0 to 29. "
            "Each index runs one CV seed "
            "containing all five folds."
        ),
    )

    group.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Aggregate all 30 per-seed "
            "text files into "
            "final_results.txt."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    if args.aggregate:

        aggregate_results()

    else:

        run_seed(
            args.seed_index
        )


if __name__ == "__main__":

    main()