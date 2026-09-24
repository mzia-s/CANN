#!/usr/bin/env python3

import numpy as np

from config import (
    RESULT_DIR,
    CV_SEEDS,
    NUMBER_OF_FOLDS,
    METHODS,
    METRIC_NAMES,
)


def progress_print(*args):

    print(
        *args,
        flush=True,
    )


# ============================================================
# TEXT FORMATTING
# ============================================================

def metrics_to_text(
    metrics,
):

    return " ".join(

        f"{name}="
        f"{metrics[name]:.10f}"

        for name
        in METRIC_NAMES
    )


def print_fold_result_line(
    seed,
    fold,
    method,
    metrics,
    cnn_epoch,
    transformer_epoch,
):

    progress_print(

        "FOLD_RESULT\t"

        f"seed={seed}\t"

        f"fold={fold}\t"

        f"method={method}\t"

        f"cnn_epoch={cnn_epoch}\t"

        f"transformer_epoch="
        f"{transformer_epoch}\t"

        +
        "\t".join(

            f"{name}="
            f"{metrics[name]:.10f}"

            for name
            in METRIC_NAMES
        )
    )


def print_oof_result_line(
    seed,
    method,
    metrics,
):

    progress_print(

        "OOF_RESULT\t"

        f"seed={seed}\t"

        f"method={method}\t"

        +
        "\t".join(

            f"{name}="
            f"{metrics[name]:.10f}"

            for name
            in METRIC_NAMES
        )
    )


# ============================================================
# RESULT FILE PARSER
# ============================================================

def parse_key_value_tokens(
    line,
):

    parts = (
        line
        .rstrip(
            "\n"
        )
        .split(
            "\t"
        )
    )

    result = {
        "record_type":
            parts[
                0
            ]
    }

    for token in parts[
        1:
    ]:

        if "=" not in token:

            continue

        (
            key,
            value,
        ) = token.split(
            "=",
            1,
        )

        result[
            key
        ] = value

    return result


def parse_seed_file(
    path,
):

    fold_results = []

    oof_results = []

    with path.open(
        "r"
    ) as handle:

        for line in handle:

            if line.startswith(
                "FOLD_RESULT\t"
            ):

                record = (
                    parse_key_value_tokens(
                        line
                    )
                )

                fold_results.append(
                    record
                )

            elif line.startswith(
                "OOF_RESULT\t"
            ):

                record = (
                    parse_key_value_tokens(
                        line
                    )
                )

                oof_results.append(
                    record
                )

    return (
        fold_results,
        oof_results,
    )


# ============================================================
# AGGREGATE ALL 30 SEEDS
# ============================================================

def aggregate_results():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        RESULT_DIR
        /
        "final_results.txt"
    )

    expected_seed_files = [

        RESULT_DIR
        /
        f"seed_{seed:02d}.txt"

        for seed
        in CV_SEEDS
    ]

    missing = [

        str(
            path
        )

        for path
        in expected_seed_files

        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Cannot aggregate because "
            "these seed files are missing:\n"
            +
            "\n".join(
                missing
            )
        )

    all_fold_records = []

    all_oof_records = []

    for path in expected_seed_files:

        (
            fold_records,
            oof_records,
        ) = parse_seed_file(
            path
        )

        expected_fold_lines = (
            NUMBER_OF_FOLDS
            *
            len(
                METHODS
            )
        )

        if (
            len(
                fold_records
            )
            !=
            expected_fold_lines
        ):

            raise RuntimeError(
                "Unexpected number of "
                "FOLD_RESULT lines in "
                f"{path}.\n"
                f"Expected "
                f"{expected_fold_lines}, "
                f"found "
                f"{len(fold_records)}."
            )

        if (
            len(
                oof_records
            )
            !=
            len(
                METHODS
            )
        ):

            raise RuntimeError(
                "Unexpected number of "
                "OOF_RESULT lines in "
                f"{path}.\n"
                f"Expected "
                f"{len(METHODS)}, "
                f"found "
                f"{len(oof_records)}."
            )

        all_fold_records.extend(
            fold_records
        )

        all_oof_records.extend(
            oof_records
        )


    # ========================================================
    # CONVERT OOF RECORDS
    # ========================================================

    typed_oof = []

    for record in all_oof_records:

        row = {

            "seed":
                int(
                    record[
                        "seed"
                    ]
                ),

            "method":
                record[
                    "method"
                ],
        }

        for metric_name in METRIC_NAMES:

            row[
                metric_name
            ] = float(
                record[
                    metric_name
                ]
            )

        typed_oof.append(
            row
        )


    # ========================================================
    # CONVERT FOLD RECORDS
    # ========================================================

    typed_fold = []

    for record in all_fold_records:

        row = {

            "seed":
                int(
                    record[
                        "seed"
                    ]
                ),

            "fold":
                int(
                    record[
                        "fold"
                    ]
                ),

            "method":
                record[
                    "method"
                ],

            "cnn_epoch":
                int(
                    record[
                        "cnn_epoch"
                    ]
                ),

            "transformer_epoch":
                int(
                    record[
                        "transformer_epoch"
                    ]
                ),
        }

        for metric_name in METRIC_NAMES:

            row[
                metric_name
            ] = float(
                record[
                    metric_name
                ]
            )

        typed_fold.append(
            row
        )


    # ========================================================
    # WRITE FINAL TEXT FILE
    # ========================================================

    with final_path.open(
        "w",
        buffering=1,
    ) as handle:

        print(
            "=" * 150,
            file=handle,
        )

        print(
            "FINAL 30-SEED x 5-FOLD "
            "CROSS-VALIDATION RESULTS",
            file=handle,
        )

        print(
            "=" * 150,
            file=handle,
        )

        print(
            "",
            file=handle,
        )

        print(
            "Feature set: "
            "PSRT facet k=3,4,5 + "
            "ordinary k-mer k=6,7",
            file=handle,
        )

        print(
            "CNN consensus weight: 0.50",
            file=handle,
        )

        print(
            "Transformer consensus weight: 0.50",
            file=handle,
        )

        print(
            "Outer CV: "
            "StratifiedKFold("
            "n_splits=5, "
            "shuffle=True)",
            file=handle,
        )

        print(
            "Within each outer development portion: "
            "approximately 70% train + "
            "10% validation",
            file=handle,
        )

        print(
            "Checkpoint: encoder cosine "
            "1-NN validation ACC; "
            "macro-F1 tie-break",
            file=handle,
        )

        print(
            "Final fold reference: "
            "train + validation",
            file=handle,
        )

        print(
            "Final statistics: "
            "mean +/- sample SD across "
            "30 seed-level OOF results",
            file=handle,
        )

        print(
            "1-NN and 5-NN are reported separately.",
            file=handle,
        )

        print(
            "",
            file=handle,
        )


        # ====================================================
        # MEAN +/- SD ACROSS 30 SEEDS
        # ====================================================

        print(
            "=" * 150,
            file=handle,
        )

        print(
            "OVERALL MEAN +/- SD "
            "ACROSS 30 SEEDS",
            file=handle,
        )

        print(
            "=" * 150,
            file=handle,
        )

        for method in METHODS:

            rows = [

                row

                for row
                in typed_oof

                if row[
                    "method"
                ]
                ==
                method
            ]

            if (
                len(
                    rows
                )
                !=
                len(
                    CV_SEEDS
                )
            ):

                raise RuntimeError(
                    f"Expected "
                    f"{len(CV_SEEDS)} "
                    "seed-level OOF rows "
                    f"for {method}, "
                    f"found {len(rows)}."
                )

            print(
                f"\n{method}",
                file=handle,
            )

            for metric_name in METRIC_NAMES:

                values = np.asarray(
                    [
                        row[
                            metric_name
                        ]

                        for row
                        in rows
                    ],
                    dtype=np.float64,
                )

                mean = (
                    values.mean()
                )

                sd = (
                    values.std(
                        ddof=1
                    )
                )

                print(
                    f"  {metric_name:<9s}: "
                    f"{mean:.6f} +/- "
                    f"{sd:.6f}",
                    file=handle,
                )


        # ====================================================
        # SEED-LEVEL OOF RESULTS
        # ====================================================

        print(
            "\n"
            +
            "=" * 150,
            file=handle,
        )

        print(
            "SEED-LEVEL COMPLETE OOF RESULTS",
            file=handle,
        )

        print(
            "=" * 150,
            file=handle,
        )

        for seed in CV_SEEDS:

            print(
                f"\nSEED {seed}",
                file=handle,
            )

            rows = [

                row

                for row
                in typed_oof

                if row[
                    "seed"
                ]
                ==
                seed
            ]

            method_to_row = {

                row[
                    "method"
                ]:
                    row

                for row
                in rows
            }

            for method in METHODS:

                row = method_to_row.get(
                    method
                )

                if row is None:

                    raise RuntimeError(
                        "Missing seed-level "
                        "OOF result: "
                        f"seed={seed}, "
                        f"method={method}"
                    )

                metric_text = " ".join(

                    f"{metric_name}="
                    f"{row[metric_name]:.6f}"

                    for metric_name
                    in METRIC_NAMES
                )

                print(
                    f"  {method:18s} | "
                    f"{metric_text}",
                    file=handle,
                )


        # ====================================================
        # ALL FOLD RESULTS
        # ====================================================

        print(
            "\n"
            +
            "=" * 150,
            file=handle,
        )

        print(
            "ALL FOLD-LEVEL TEST RESULTS",
            file=handle,
        )

        print(
            "=" * 150,
            file=handle,
        )

        for seed in CV_SEEDS:

            print(
                f"\nSEED {seed}",
                file=handle,
            )

            for fold in range(
                1,
                NUMBER_OF_FOLDS + 1,
            ):

                fold_rows = [

                    row

                    for row
                    in typed_fold

                    if (
                        row[
                            "seed"
                        ]
                        ==
                        seed

                        and

                        row[
                            "fold"
                        ]
                        ==
                        fold
                    )
                ]

                if (
                    len(
                        fold_rows
                    )
                    !=
                    len(
                        METHODS
                    )
                ):

                    raise RuntimeError(
                        "Missing fold results: "
                        f"seed={seed}, "
                        f"fold={fold}"
                    )

                first = (
                    fold_rows[
                        0
                    ]
                )

                print(
                    f"  Fold {fold} | "
                    f"CNN epoch="
                    f"{first['cnn_epoch']} | "
                    f"Transformer epoch="
                    f"{first['transformer_epoch']}",
                    file=handle,
                )

                method_to_row = {

                    row[
                        "method"
                    ]:
                        row

                    for row
                    in fold_rows
                }

                for method in METHODS:

                    row = (
                        method_to_row[
                            method
                        ]
                    )

                    metric_text = " ".join(

                        f"{metric_name}="
                        f"{row[metric_name]:.6f}"

                        for metric_name
                        in METRIC_NAMES
                    )

                    print(
                        f"    {method:18s} | "
                        f"{metric_text}",
                        file=handle,
                    )


        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print(
            "\n"
            +
            "=" * 150,
            file=handle,
        )

        print(
            "SELECTED EPOCH SUMMARY "
            "ACROSS ALL 150 FOLDS",
            file=handle,
        )

        print(
            "=" * 150,
            file=handle,
        )

        epoch_rows = [

            row

            for row
            in typed_fold

            if row[
                "method"
            ]
            ==
            "CNN 1-NN"
        ]

        cnn_epochs = np.asarray(
            [
                row[
                    "cnn_epoch"
                ]

                for row
                in epoch_rows
            ],
            dtype=np.float64,
        )

        transformer_epochs = np.asarray(
            [
                row[
                    "transformer_epoch"
                ]

                for row
                in epoch_rows
            ],
            dtype=np.float64,
        )

        print(
            "CNN selected epoch: "
            f"mean={cnn_epochs.mean():.3f}, "
            f"SD={cnn_epochs.std(ddof=1):.3f}, "
            f"min={int(cnn_epochs.min())}, "
            f"max={int(cnn_epochs.max())}",
            file=handle,
        )

        print(
            "Transformer selected epoch: "
            f"mean={transformer_epochs.mean():.3f}, "
            f"SD={transformer_epochs.std(ddof=1):.3f}, "
            f"min={int(transformer_epochs.min())}, "
            f"max={int(transformer_epochs.max())}",
            file=handle,
        )

    print(
        "Final aggregation written to:",
        final_path,
    )