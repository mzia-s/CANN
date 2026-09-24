#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUMBER_OF_FILTRATION_POSITIONS,
    TRANSFORMER_DIMENSION,
    TRANSFORMER_LAYERS,
    TRANSFORMER_HEADS,
    TRANSFORMER_DROPOUT,
    PROJECTION_DIMENSION,
)


# ============================================================
# TRANSFORMER LAYER
# ============================================================

class RegularTransformerLayer(
    nn.Module
):

    def __init__(
        self,
        dim,
        heads,
        dropout,
    ):

        super().__init__()

        self.heads = (
            heads
        )

        self.head_dimension = (
            dim
            //
            heads
        )

        self.W_q = nn.Linear(
            dim,
            dim,
        )

        self.W_k = nn.Linear(
            dim,
            dim,
        )

        self.W_v = nn.Linear(
            dim,
            dim,
        )

        self.W_o = nn.Linear(
            dim,
            dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                dim,
                dim * 4,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim * 4,
                dim,
            ),
        )


    def forward(
        self,
        x,
    ):

        (
            batch_size,
            token_count,
            dimension,
        ) = x.shape

        q = (
            self.W_q(
                x
            )
            .view(
                batch_size,
                token_count,
                self.heads,
                self.head_dimension,
            )
            .transpose(
                1,
                2,
            )
        )

        k = (
            self.W_k(
                x
            )
            .view(
                batch_size,
                token_count,
                self.heads,
                self.head_dimension,
            )
            .transpose(
                1,
                2,
            )
        )

        v = (
            self.W_v(
                x
            )
            .view(
                batch_size,
                token_count,
                self.heads,
                self.head_dimension,
            )
            .transpose(
                1,
                2,
            )
        )

        scores = (
            torch.matmul(
                q,
                k.transpose(
                    -2,
                    -1,
                ),
            )
            /
            (
                self.head_dimension
                **
                0.5
            )
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        output = torch.matmul(
            self.dropout(
                attention
            ),
            v,
        )

        output = (
            output
            .transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                token_count,
                dimension,
            )
        )

        output = (
            self.W_o(
                output
            )
        )

        x = self.norm1(
            x
            +
            self.dropout(
                output
            )
        )

        ffn_output = (
            self.ffn(
                x
            )
        )

        return self.norm2(
            x
            +
            self.dropout(
                ffn_output
            )
        )


# ============================================================
# TRANSFORMER STACK
# ============================================================

class RegularTransformer(
    nn.Module
):

    def __init__(
        self,
        dim,
        num_layers,
        heads,
        dropout,
    ):

        super().__init__()

        self.layers = nn.ModuleList(
            [

                RegularTransformerLayer(
                    dim,
                    heads,
                    dropout,
                )

                for _
                in range(
                    num_layers
                )
            ]
        )


    def forward(
        self,
        x,
    ):

        for layer in self.layers:

            x = layer(
                x
            )

        return x


# ============================================================
# HYBRID TRANSFORMER
# ============================================================

class HybridFacetKmerTransformer(
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

        projections = {}

        for index, (
            modality,
            k,
        ) in enumerate(
            self.input_specs
        ):

            if modality == "facet":

                input_dimension = (
                    (
                        4 ** k
                    )
                    *
                    NUMBER_OF_FILTRATION_POSITIONS
                )

            else:

                input_dimension = (
                    4 ** k
                )

            name = (
                f"input_{index}_{modality}_k{k}"
            )

            projections[
                name
            ] = nn.Linear(
                input_dimension,
                TRANSFORMER_DIMENSION,
            )

        self.projection_names = list(
            projections.keys()
        )

        self.token_projections = nn.ModuleDict(
            projections
        )

        self.position_embedding = nn.Parameter(

            torch.zeros(
                1,
                len(
                    self.input_specs
                ),
                TRANSFORMER_DIMENSION,
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        self.input_dropout = nn.Dropout(
            TRANSFORMER_DROPOUT
        )

        self.transformer = RegularTransformer(
            TRANSFORMER_DIMENSION,
            TRANSFORMER_LAYERS,
            TRANSFORMER_HEADS,
            TRANSFORMER_DROPOUT,
        )

        self.encoder_bn = nn.BatchNorm1d(
            TRANSFORMER_DIMENSION
        )

        self.classifier_dropout = nn.Dropout(
            0.30
        )

        self.classifier = nn.Linear(
            TRANSFORMER_DIMENSION,
            num_classes,
        )

        self.projection_head = nn.Sequential(

            nn.Linear(
                TRANSFORMER_DIMENSION,
                TRANSFORMER_DIMENSION,
            ),

            nn.ReLU(),

            nn.Linear(
                TRANSFORMER_DIMENSION,
                PROJECTION_DIMENSION,
            ),
        )


    def encode(
        self,
        blocks,
    ):

        tokens = []

        for block, name in zip(
            blocks,
            self.projection_names,
        ):

            flattened = torch.flatten(
                block,
                start_dim=1,
            )

            tokens.append(
                self.token_projections[
                    name
                ](
                    flattened
                )
            )

        x = torch.stack(
            tokens,
            dim=1,
        )

        x = (
            x
            +
            self.position_embedding
        )

        x = self.input_dropout(
            x
        )

        x = self.transformer(
            x
        )

        embedding = x.mean(
            dim=1
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