#!/usr/bin/env python3

import math

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
)

from config import (
    INPUT_SPECS,
    MAX_FAMILIES_PER_BATCH,
    GENOMES_PER_FAMILY,
    EVALUATION_BATCH_SIZE,
    NUM_WORKERS,
)


# ============================================================
# HYBRID DATASET
# ============================================================

class HybridDataset(
    Dataset
):

    def __init__(
        self,
        facet_data,
        kmer_data,
        labels,
        indices,
        input_specs,
    ):

        self.facet_data = (
            facet_data
        )

        self.kmer_data = (
            kmer_data
        )

        self.labels = (
            labels
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.input_specs = list(
            input_specs
        )


    def __len__(
        self,
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        local_index,
    ):

        global_index = (
            self.indices[
                local_index
            ]
        )

        blocks = []

        for (
            modality,
            k,
        ) in self.input_specs:

            if modality == "facet":

                array = np.ascontiguousarray(
                    self.facet_data[
                        k
                    ][
                        global_index
                    ].T,
                    dtype=np.float32,
                )

            else:

                array = np.ascontiguousarray(
                    self.kmer_data[
                        k
                    ][
                        global_index
                    ][
                        None,
                        :
                    ],
                    dtype=np.float32,
                )

            blocks.append(
                torch.from_numpy(
                    array
                )
            )

        label = torch.tensor(
            self.labels[
                global_index
            ],
            dtype=torch.long,
        )

        return (
            blocks,
            label,
        )


# ============================================================
# BALANCED FAMILY SAMPLER
# ============================================================

class BalancedFamilyBatchSampler(
    Sampler
):

    def __init__(
        self,
        labels,
        families_per_batch,
        genomes_per_family,
        batches_per_epoch,
        seed,
    ):

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        self.families_per_batch = int(
            families_per_batch
        )

        self.genomes_per_family = int(
            genomes_per_family
        )

        self.batches_per_epoch = int(
            batches_per_epoch
        )

        self.seed = int(
            seed
        )

        self.epoch = 0

        self.unique_families = np.unique(
            self.labels
        )

        self.indices_by_family = {

            family:
                np.where(
                    self.labels
                    ==
                    family
                )[0]

            for family
            in self.unique_families
        }


    def __len__(
        self,
    ):

        return (
            self.batches_per_epoch
        )


    def __iter__(
        self,
    ):

        rng = np.random.default_rng(
            self.seed
            +
            self.epoch
        )

        self.epoch += 1

        for _ in range(
            self.batches_per_epoch
        ):

            selected_families = rng.choice(
                self.unique_families,
                size=self.families_per_batch,
                replace=False,
            )

            batch_indices = []

            for family in selected_families:

                available = (
                    self.indices_by_family[
                        family
                    ]
                )

                selected = rng.choice(
                    available,
                    size=self.genomes_per_family,
                    replace=(
                        len(
                            available
                        )
                        <
                        self.genomes_per_family
                    ),
                )

                batch_indices.extend(
                    selected.tolist()
                )

            rng.shuffle(
                batch_indices
            )

            yield batch_indices


# ============================================================
# BUILD FOLD LOADERS
# ============================================================

def build_fold_loaders(
    facet_data,
    kmer_data,
    y,
    train_indices,
    validation_indices,
    reference_indices,
    test_indices,
    sampler_seed,
):

    train_dataset = HybridDataset(
        facet_data,
        kmer_data,
        y,
        train_indices,
        INPUT_SPECS,
    )

    validation_dataset = HybridDataset(
        facet_data,
        kmer_data,
        y,
        validation_indices,
        INPUT_SPECS,
    )

    reference_dataset = HybridDataset(
        facet_data,
        kmer_data,
        y,
        reference_indices,
        INPUT_SPECS,
    )

    test_dataset = HybridDataset(
        facet_data,
        kmer_data,
        y,
        test_indices,
        INPUT_SPECS,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    reference_loader = DataLoader(
        reference_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    training_family_count = len(
        np.unique(
            y[
                train_indices
            ]
        )
    )

    families_per_batch = min(
        MAX_FAMILIES_PER_BATCH,
        training_family_count,
    )

    balanced_batch_size = (
        families_per_batch
        *
        GENOMES_PER_FAMILY
    )

    batches_per_epoch = math.ceil(
        len(
            train_indices
        )
        /
        balanced_batch_size
    )

    return {

        "train_dataset":
            train_dataset,

        "train_labels_for_sampler":
            y[
                train_indices
            ].copy(),

        "families_per_batch":
            families_per_batch,

        "batches_per_epoch":
            batches_per_epoch,

        "sampler_seed":
            sampler_seed,

        "train_eval_loader":
            train_eval_loader,

        "validation_loader":
            validation_loader,

        "reference_loader":
            reference_loader,

        "test_loader":
            test_loader,
    }


# ============================================================
# CREATE FRESH TRAINING LOADER
# ============================================================

def create_training_loader(
    loaders,
):

    sampler = BalancedFamilyBatchSampler(
        labels=(
            loaders[
                "train_labels_for_sampler"
            ]
        ),
        families_per_batch=(
            loaders[
                "families_per_batch"
            ]
        ),
        genomes_per_family=(
            GENOMES_PER_FAMILY
        ),
        batches_per_epoch=(
            loaders[
                "batches_per_epoch"
            ]
        ),
        seed=(
            loaders[
                "sampler_seed"
            ]
        ),
    )

    return DataLoader(
        loaders[
            "train_dataset"
        ],
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
    )