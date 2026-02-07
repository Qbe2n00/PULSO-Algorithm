# PULSO-Algorithm
PULSO: Predictive Upstream Ligands using Single-cell transcriptOmics

![PULSO-github](https://github.com/user-attachments/assets/e610cbab-e665-41d0-b948-f4185db72999)

## Method Overview
Although single-cell RNA sequencing (scRNA-seq) has revolutionized our ability to dissect complex biological processes by resolving cell-cell interactions at unprecedented resolution, existing computational frameworks often lose cell-level granularity through bulk aggregation and rely on gene-level inference that is vulnerable to high dropout rates. To overcome these limitations, we developed PULSO (Predictive Upstream Ligands using Single-cell transcriptOmics) — a computational framework that infers upstream ligand signaling activity at the single-cell level directly from transcriptomic profiles.

## Ligand Activity Inference
PULSO refines ligand activity quantification by integrating [SIMBA](https://github.com/pinellolab/simba)-based cell–gene co-embedding with NicheNet to infer context-specific signaling based on embedding distances rather than raw gene expression values. After filtering ligand–target matrices to retain only expressed features, SIMBA co-embedding was performed with “sample” as a batch key, where gene expression was discretized into five k-means bins and cell–cell edges were constructed using Euclidean distance in a 30-dimensional HVG space. The resulting joint embeddings were used to compute normalized pairwise cell–gene distances, which replaced differential expression values as NicheNet inputs. Ligand activity per cell was estimated as the Pearson correlation between normalized distances and NicheNet regulatory potential scores, retaining only ligands with cognate receptor expression in target populations. Ligands with positive activity in at least 10 cells were selected for downstream analyses.

## Cell-Cell Interaction Analysis
To reconstruct cell–cell communication networks, PULSO incorporated significant ligand candidates into a liana-based interaction analysis using OmniPath ligand–receptor annotations. For each major cell type, ligands expressed in sender populations and receptors expressed in corresponding receiver subtypes were evaluated, with significance defined by geometric mean interaction scores at P < 0.01. Network visualization was performed using normalized edge weights to represent cumulative ligand–receptor activity among condition-specific sender–receiver pairs.
