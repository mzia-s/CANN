#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CNN_BRANCH_DROPOUT,
    CNN_ENCODER_DIMENSION,
    PROJECTION_DIMENSION,
)


# ============================================================
# CNN RESIDUAL BLOCK
# ============================================================

class ResidualConvBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
    ):

        super().__init__()

        padding = (
            kernel_size
            //
            2
        )

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm1d(
                    out_channels
                ),
            )

        else:

            self.shortcut = (
                nn.Identity()
            )


    def forward(
        self,
        x,
    ):

        residual = (
            self.shortcut(
                x
            )
        )

        x = F.relu(
            self.bn1(
                self.conv1(
                    x
                )
            )
        )

        x = self.bn2(
            self.conv2(
                x
            )
        )

        return F.relu(
            x
            +
            residual
        )


# ============================================================
# CNN BRANCH
# ============================================================

class HybridCNNBranch(
    nn.Module
):

    def __init__(
        self,
        input_channels,
    ):

        super().__init__()

        self.initial_conv = nn.Conv1d(
            input_channels,
            16,
            kernel_size=7,
            stride=4,
            padding=3,
            bias=False,
        )

        self.initial_bn = nn.BatchNorm1d(
            16
        )

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        self.residual = ResidualConvBlock(
            16,
            32,
            kernel_size=5,
        )

        self.dropout = nn.Dropout1d(
            CNN_BRANCH_DROPOUT
        )

        self.downsample = nn.Conv1d(
            32,
            32,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False,
        )

        self.downsample_bn = nn.BatchNorm1d(
            32
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(
            4
        )


    def forward(
        self,
        x,
    ):

        x = F.relu(
            self.initial_bn(
                self.initial_conv(
                    x
                )
            )
        )

        x = self.pool(
            x
        )

        x = self.residual(
            x
        )

        x = self.dropout(
            x
        )

        x = F.relu(
            self.downsample_bn(
                self.downsample(
                    x
                )
            )
        )

        x = self.adaptive_pool(
            x
        )

        return torch.flatten(
            x,
            start_dim=1,
        )


# ============================================================
# HYBRID CNN
# ============================================================

class HybridFacetKmerCNN(
    nn.Module
):

    def __init__(
        self,
        input_specs,
        num_classes,
    ):

        super().__init__()

        self.input_specs = list(
            input_specs
        )

        branches = {}

        for index, (
            modality,
            k,
        ) in enumerate(
            self.input_specs
        ):

            input_channels = (
                2
                if modality
                ==
                "facet"
                else
                1
            )

            name = (
                f"input_{index}_{modality}_k{k}"
            )

            branches[
                name
            ] = HybridCNNBranch(
                input_channels
            )

        self.branch_names = list(
            branches.keys()
        )

        self.branches = nn.ModuleDict(
            branches
        )

        combined_dimension = (
            len(
                self.input_specs
            )
            *
            128
        )

        self.encoder_layer = nn.Linear(
            combined_dimension,
            CNN_ENCODER_DIMENSION,
            bias=False,
        )

        self.encoder_bn = nn.BatchNorm1d(
            CNN_ENCODER_DIMENSION
        )

        self.classifier_dropout = nn.Dropout(
            0.30
        )

        self.classifier = nn.Linear(
            CNN_ENCODER_DIMENSION,
            num_classes,
        )

        self.projection_head = nn.Sequential(

            nn.Linear(
                CNN_ENCODER_DIMENSION,
                CNN_ENCODER_DIMENSION,
            ),

            nn.ReLU(),

            nn.Linear(
                CNN_ENCODER_DIMENSION,
                PROJECTION_DIMENSION,
            ),
        )


    def encode(
        self,
        blocks,
    ):

        outputs = []

        for block, name in zip(
            blocks,
            self.branch_names,
        ):

            outputs.append(
                self.branches[
                    name
                ](
                    block
                )
            )

        combined = torch.cat(
            outputs,
            dim=1,
        )

        embedding = (
            self.encoder_layer(
                combined
            )
        )

        return self.encoder_bn(
            embedding
        )


    def project(
        self,
        embedding,
    ):

        return F.normalize(
            self.projection_head(
                embedding
            ),
            p=2,
            dim=1,
        )


    def forward(
        self,
        blocks,
    ):

        embedding = self.encode(
            blocks
        )

        logits = self.classifier(
            self.classifier_dropout(
                embedding
            )
        )

        projection = self.project(
            embedding
        )

        return (
            logits,
            embedding,
            projection,
        )