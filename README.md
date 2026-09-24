# CANN

Alignment-free viral family classification using commutative-algebraic descriptors and neural representation learning.

## Introduction

Viral family classification requires sequence representations that capture both nucleotide composition and the positional organization of genomic patterns. CANN combines persistent facet-number descriptors for k-mers of lengths 3, 4, and 5 with complementary k-mer frequency features for lengths 6 and 7.

The five feature blocks are processed by independently trained convolutional neural network (CNN) and Transformer encoders. Cosine distances between the resulting genome embeddings are averaged with equal weights to obtain consensus distances for nearest-neighbor classification. Consensus 1-NN is the principal prediction method, while 5-NN provides an additional evaluation.

The framework builds on the commutative-algebraic sequence representations introduced in [CAKR: commutative algebra k-mer representations for genomics](https://doi.org/10.1038/s41467-026-76429-z), extending their use through supervised neural representation learning and consensus genome comparison.

The main benchmarks use stratified five-fold cross-validation repeated over 30 random seeds. Held-out predictions are pooled across the five folds within each seed, and performance is reported as the mean and sample standard deviation across seeds.

## CANN Workflow

[![Overview of the CANN workflow](figures/CANN_workflow.png)](figures/workflow.pdf)

**Figure:** Overview of CANN, including sequence feature generation, independent CNN and Transformer training, consensus distance construction, and nearest-neighbor classification under repeated stratified five-fold cross-validation.

[View the workflow as a PDF](figures/workflow.pdf)

## Data Sources and Availability

The genomic data supporting this study were obtained from the National Center for Biotechnology Information (NCBI), including GenBank and, where applicable, the NCBI Virus resource.

Sequence records can be accessed and downloaded through:

- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
- [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/)

The metadata files identify the accession versions and viral family labels used for each dataset. Use the listed accession versions and the supplied family labels when reconstructing the benchmarks, because database sequences and taxonomic assignments may change over time.

### Dataset Files

| Dataset | Metadata CSV | Corresponding FASTA |
| --- | --- | --- |
| NCBI 2020 | `Yau2020_record_processed.csv` | `Yau2020_record_processed.fasta` |
| NCBI 2022 | `Yau2022_record_processed.csv` | `Yau2022_record_processed.fasta` |
| NCBI 2024 | `NCBI_record_valid_nucleotide.csv` | `NCBI_record_valid_nucleotide.fasta` |
| NCBI 2024 All | `NCBI_record_valid_count.csv` | `NCBI_record_valid_count.fasta` |

Place each FASTA file beside its corresponding CSV in the appropriate subfolder of `datasets/`.

The CSV files contain metadata and classification labels. The corresponding FASTA files are required for sequence preprocessing and feature generation.

## Acknowledgments

We acknowledge Dr. Faisal Suwayyid and collaborators for developing CAKR and making its implementation publicly available. Their work on commutative-algebraic k-mer representations provides a foundation for the algebraic sequence descriptors used in CANN.

- **Paper:** Suwayyid, F., Hozumi, Y., Zia, M., Wee, J., Feng, H., and Wei, G.-W. *CAKR: commutative algebra k-mer representations for genomics*. Nature Communications **17**, 9644 (2026).
- **DOI:** https://doi.org/10.1038/s41467-026-76429-z
- **Code:** https://github.com/FaisalSuwayyid/CAKL

The repository retains the earlier directory name `CAKL`; the published framework is named **CAKR**.

We also retain the acknowledgment of [KmerTopology](https://github.com/hozumiyu/KmerTopology) by Yuta Hozumi included in the supplied `psrt.py`.
