# MMOAE_code
Example code of MMOAE

# Multi-omics Deep Learning Framework

This repository implements a deep representation learning pipeline for multi-omics cancer data.

It covers three primary tasks:

## 🔬 Task 1: Representation Learning & Visualization
Unsupervised pretraining of a multi-modal autoencoder (MMOAE) on TCGA pan-cancer multi-omics data, followed by clustering and t-SNE visualization.

- Input: `multiomics.csv`
- Output: Latent features, t-SNE plots, clustering scores

## 🧬 Task 2: Cancer Subtype Classification (e.g. BRCA)
Fine-tune the pretrained encoder on breast cancer (BRCA) samples with known subtypes (PAM50 labels). Includes evaluation with SVM classifiers.

- Input: TCGA-BRCA multiomics + `tcgaphenotype/tcgabrcaphonetype.csv`
- Output: Trained classifier, classification report (CSV), feature embeddings

## 🧠 Task 3: Survival Prediction (e.g. COADREAD)
Fine-tune the pretrained encoder for downstream survival prediction (OS). Includes Cox loss, risk prediction, clustering, and Kaplan-Meier plots.

- Input: COADREAD subset of `multiomics.csv` + `survival.csv`
- Output: Risk scores, C-index, log-rank p-values, survival curves
