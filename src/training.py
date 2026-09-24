#!/usr/bin/env python3

import random

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.neighbors import KNeighborsClassifier

from config import (
    CHECKPOINT_TOLERANCE,
    PRIMARY_SELECTION_METHOD,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MINIMUM_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    CONTRASTIVE_WEIGHT,
    CONTRASTIVE_TEMPERATURE,
)

from dataset import (
    create_training_loader,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def progress_print(*args):

    print(
        *args,
        flush=True,
    )


def set_global_seed(seed):

    np.random.seed(
        seed
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    try:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    except AttributeError:

        pass


def make_run_seed(
    cv_seed,
    fold_number,
):

    return int(
        cv_seed * 1000
        +
        fold_number
    )


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

def supervised_contrastive_loss(
    projections,
    labels,
    temperature=0.07,
):

    batch_size = projections.size(
        0
    )

    labels = labels.view(
        -1,
        1,
    )

    same_family = (
        labels
        ==
        labels.T
    ).float()

    identity = torch.eye(
        batch_size,
        device=projections.device,
    )

    positive_mask = (
        same_family
        *
        (
            1.0
            -
            identity
        )
    )

    similarities = (
        torch.matmul(
            projections,
            projections.T,
        )
        /
        temperature
    )

    similarities = (
        similarities
        -
        similarities.max(
            dim=1,
            keepdim=True,
        ).values.detach()
    )

    nonself = (
        1.0
        -
        identity
    )

    exponential = (
        torch.exp(
            similarities
        )
        *
        nonself
    )

    log_probabilities = (
        similarities
        -
        torch.log(
            exponential.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-12
        )
    )

    positive_counts = (
        positive_mask.sum(
            dim=1
        )
    )

    valid = (
        positive_counts
        >
        0
    )

    mean_positive = (
        (
            positive_mask
            *
            log_probabilities
        )
        .sum(
            dim=1
        )
        /
        positive_counts.clamp_min(
            1.0
        )
    )

    return (
        -mean_positive[
            valid
        ]
        .mean()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predictions,
):

    return {

        "ACC":
            accuracy_score(
                true_labels,
                predictions,
            ),

        "BA":
            balanced_accuracy_score(
                true_labels,
                predictions,
            ),

        "F1":
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "Recall":
            recall_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "Precision":
            precision_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
    }


# ============================================================
# COSINE KNN
# ============================================================

def cosine_knn_predict(
    reference_embeddings,
    reference_labels,
    query_embeddings,
    neighbors,
):

    classifier = KNeighborsClassifier(
        n_neighbors=neighbors,
        metric="cosine",
        algorithm="brute",
        weights="uniform",
        n_jobs=-1,
    )

    classifier.fit(
        reference_embeddings,
        reference_labels,
    )

    return classifier.predict(
        query_embeddings
    )


# ============================================================
# DEVICE HELPER
# ============================================================

def move_blocks_to_device(
    blocks,
    device,
):

    return [

        block.to(
            device
        )

        for block
        in blocks
    ]


# ============================================================
# EXTRACT ENCODER EMBEDDINGS
# ============================================================

def extract_encoder_embeddings(
    model,
    loader,
    device,
):

    model.eval()

    embeddings = []

    labels = []

    with torch.no_grad():

        for blocks, y_batch in loader:

            blocks = (
                move_blocks_to_device(
                    blocks,
                    device,
                )
            )

            embedding = model.encode(
                blocks
            )

            embeddings.append(
                embedding.cpu()
            )

            labels.append(
                y_batch.cpu()
            )

    return (

        torch.cat(
            embeddings,
            dim=0,
        )
        .numpy()
        .astype(
            np.float32,
            copy=False,
        ),

        torch.cat(
            labels,
            dim=0,
        )
        .numpy()
        .astype(
            np.int64,
            copy=False,
        ),
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def checkpoint_improved(
    metrics,
    best_accuracy,
    best_f1,
):

    if (
        metrics[
            "ACC"
        ]
        >
        best_accuracy
        +
        CHECKPOINT_TOLERANCE
    ):

        return True

    if (
        abs(
            metrics[
                "ACC"
            ]
            -
            best_accuracy
        )
        <=
        CHECKPOINT_TOLERANCE

        and

        metrics[
            "F1"
        ]
        >
        best_f1
        +
        CHECKPOINT_TOLERANCE
    ):

        return True

    return False


def clone_state(
    model,
):

    return {

        name:
            value
            .detach()
            .cpu()
            .clone()

        for name, value
        in model.state_dict().items()
    }


# ============================================================
# VALIDATION EVALUATION
# ============================================================

def evaluate_validation(
    model,
    train_eval_loader,
    validation_loader,
    device,
):

    (
        train_embeddings,
        train_labels,
    ) = extract_encoder_embeddings(
        model,
        train_eval_loader,
        device,
    )

    (
        validation_embeddings,
        validation_labels,
    ) = extract_encoder_embeddings(
        model,
        validation_loader,
        device,
    )

    prediction_1 = cosine_knn_predict(
        train_embeddings,
        train_labels,
        validation_embeddings,
        1,
    )

    prediction_5 = cosine_knn_predict(
        train_embeddings,
        train_labels,
        validation_embeddings,
        5,
    )

    return {

        "Encoder + Cosine 1-NN":
            calculate_metrics(
                validation_labels,
                prediction_1,
            ),

        "Encoder + Cosine 5-NN":
            calculate_metrics(
                validation_labels,
                prediction_5,
            ),
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    model_name,
    maximum_epochs,
    learning_rate,
    weight_decay,
    loaders,
    device,
):

    training_loader = (
        create_training_loader(
            loaders
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MINIMUM_LEARNING_RATE,
        )
    )

    cross_entropy = nn.CrossEntropyLoss()

    best_accuracy = -np.inf

    best_f1 = -np.inf

    best_epoch = None

    best_state = None

    best_validation_metrics = None

    epochs_without_improvement = 0

    progress_print(
        "\n"
        +
        "#" * 120
    )

    progress_print(
        f"{model_name} TRAINING"
    )

    progress_print(
        "#" * 120
    )

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for blocks, labels in training_loader:

            blocks = move_blocks_to_device(
                blocks,
                device,
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                logits,
                _,
                projections,
            ) = model(
                blocks
            )

            ce = cross_entropy(
                logits,
                labels,
            )

            supcon = supervised_contrastive_loss(
                projections,
                labels,
                CONTRASTIVE_TEMPERATURE,
            )

            loss = (
                ce
                +
                CONTRASTIVE_WEIGHT
                *
                supcon
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            batch_size = labels.size(
                0
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            correct += (
                (
                    logits.argmax(
                        dim=1
                    )
                    ==
                    labels
                )
                .sum()
                .item()
            )

            total += (
                batch_size
            )

        validation_metrics = (
            evaluate_validation(
                model,
                loaders[
                    "train_eval_loader"
                ],
                loaders[
                    "validation_loader"
                ],
                device,
            )
        )

        primary = (
            validation_metrics[
                PRIMARY_SELECTION_METHOD
            ]
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        improved = checkpoint_improved(
            primary,
            best_accuracy,
            best_f1,
        )

        if improved:

            best_accuracy = (
                primary[
                    "ACC"
                ]
            )

            best_f1 = (
                primary[
                    "F1"
                ]
            )

            best_epoch = (
                epoch
            )

            best_state = clone_state(
                model
            )

            best_validation_metrics = {

                name:
                    metrics.copy()

                for name, metrics
                in validation_metrics.items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        scheduler.step(
            primary[
                "ACC"
            ]
        )

        progress_print(

            f"Epoch {epoch:02d} | "

            f"Loss="
            f"{total_loss / total:.4f} | "

            f"Train ACC="
            f"{correct / total:.4f} | "

            f"Enc Cos1="
            f"{validation_metrics['Encoder + Cosine 1-NN']['ACC']:.4f} | "

            f"Enc Cos5="
            f"{validation_metrics['Encoder + Cosine 5-NN']['ACC']:.4f} | "

            f"BA="
            f"{primary['BA']:.4f} | "

            f"F1="
            f"{primary['F1']:.4f} | "

            f"LR="
            f"{current_lr:.6f}"
        )

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            progress_print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            f"No checkpoint selected for "
            f"{model_name}."
        )

    model.load_state_dict(
        best_state
    )

    progress_print(
        f"BEST {model_name} EPOCH:",
        best_epoch,
    )

    for method, metrics in (
        best_validation_metrics.items()
    ):

        progress_print(

            method,

            f"ACC={metrics['ACC']:.6f}",

            f"BA={metrics['BA']:.6f}",

            f"F1={metrics['F1']:.6f}",

            f"Recall={metrics['Recall']:.6f}",

            f"Precision={metrics['Precision']:.6f}",
        )

    return (
        model,
        best_epoch,
        best_validation_metrics,
    )