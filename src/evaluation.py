#!/usr/bin/env python3

import numpy as np

from sklearn.metrics.pairwise import (
    cosine_distances,
)

from config import (
    METHODS,
    CNN_CONSENSUS_WEIGHT,
    TRANSFORMER_CONSENSUS_WEIGHT,
)

from training import (
    extract_encoder_embeddings,
    cosine_knn_predict,
    calculate_metrics,
)


# ============================================================
# CONSENSUS DISTANCE KNN
# ============================================================

def predict_from_distance_matrix(
    distance_matrix,
    reference_labels,
    neighbors,
    number_of_classes,
):

    if neighbors == 1:

        nearest = np.argmin(
            distance_matrix,
            axis=1,
        )

        return reference_labels[
            nearest
        ]

    nearest = np.argpartition(
        distance_matrix,
        kth=(
            neighbors
            -
            1
        ),
        axis=1,
    )[
        :,
        :neighbors
    ]

    predictions = np.empty(
        distance_matrix.shape[
            0
        ],
        dtype=np.int64,
    )

    for row in range(
        distance_matrix.shape[
            0
        ]
    ):

        neighbor_labels = (
            reference_labels[
                nearest[
                    row
                ]
            ]
        )

        counts = np.bincount(
            neighbor_labels,
            minlength=number_of_classes,
        )

        predictions[
            row
        ] = np.argmax(
            counts
        )

    return predictions


# ============================================================
# EVALUATE ONE HELD-OUT FOLD
# ============================================================

def evaluate_fold_models(
    cnn_model,
    transformer_model,
    loaders,
    number_of_classes,
    device,
):

    (
        cnn_reference_embeddings,
        reference_labels,
    ) = extract_encoder_embeddings(
        cnn_model,
        loaders[
            "reference_loader"
        ],
        device,
    )

    (
        cnn_test_embeddings,
        test_labels,
    ) = extract_encoder_embeddings(
        cnn_model,
        loaders[
            "test_loader"
        ],
        device,
    )

    (
        transformer_reference_embeddings,
        transformer_reference_labels,
    ) = extract_encoder_embeddings(
        transformer_model,
        loaders[
            "reference_loader"
        ],
        device,
    )

    (
        transformer_test_embeddings,
        transformer_test_labels,
    ) = extract_encoder_embeddings(
        transformer_model,
        loaders[
            "test_loader"
        ],
        device,
    )

    if not np.array_equal(
        reference_labels,
        transformer_reference_labels,
    ):

        raise RuntimeError(
            "CNN and Transformer reference "
            "label order differs."
        )

    if not np.array_equal(
        test_labels,
        transformer_test_labels,
    ):

        raise RuntimeError(
            "CNN and Transformer test "
            "label order differs."
        )

    predictions = {}


    # ========================================================
    # CNN 1-NN
    # ========================================================

    predictions[
        "CNN 1-NN"
    ] = cosine_knn_predict(
        cnn_reference_embeddings,
        reference_labels,
        cnn_test_embeddings,
        1,
    )


    # ========================================================
    # CNN 5-NN
    # ========================================================

    predictions[
        "CNN 5-NN"
    ] = cosine_knn_predict(
        cnn_reference_embeddings,
        reference_labels,
        cnn_test_embeddings,
        5,
    )


    # ========================================================
    # TRANSFORMER 1-NN
    # ========================================================

    predictions[
        "Transformer 1-NN"
    ] = cosine_knn_predict(
        transformer_reference_embeddings,
        reference_labels,
        transformer_test_embeddings,
        1,
    )


    # ========================================================
    # TRANSFORMER 5-NN
    # ========================================================

    predictions[
        "Transformer 5-NN"
    ] = cosine_knn_predict(
        transformer_reference_embeddings,
        reference_labels,
        transformer_test_embeddings,
        5,
    )


    # ========================================================
    # FIXED 50/50 CONSENSUS
    # ========================================================

    cnn_test_distances = cosine_distances(
        cnn_test_embeddings,
        cnn_reference_embeddings,
    )

    transformer_test_distances = cosine_distances(
        transformer_test_embeddings,
        transformer_reference_embeddings,
    )

    fixed_test_distances = (
        CNN_CONSENSUS_WEIGHT
        *
        cnn_test_distances
        +
        TRANSFORMER_CONSENSUS_WEIGHT
        *
        transformer_test_distances
    )


    # ========================================================
    # CONSENSUS 1-NN
    # ========================================================

    predictions[
        "Consensus 1-NN"
    ] = predict_from_distance_matrix(
        fixed_test_distances,
        reference_labels,
        1,
        number_of_classes,
    )


    # ========================================================
    # CONSENSUS 5-NN
    # ========================================================

    predictions[
        "Consensus 5-NN"
    ] = predict_from_distance_matrix(
        fixed_test_distances,
        reference_labels,
        5,
        number_of_classes,
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = {

        method:
            calculate_metrics(
                test_labels,
                predictions[
                    method
                ],
            )

        for method
        in METHODS
    }

    return (
        predictions,
        metrics,
        test_labels,
    )