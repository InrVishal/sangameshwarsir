# Genomic Regulatory Elements Deep Learning Model

## Overview
This document outlines the methodology and architecture of the Deep Learning model designed to predict interactions between regulatory elements (specifically enhancers). The model learns sequential DNA motifs and their long-range dependencies from genomic data.

## Dataset
The dataset provided (`CDD (17544 ones,2924 zeros).xlsx`) consists of **20,468 DNA sequences**:
- **Positive Samples (Ones):** 17,544 sequences (Enhancers / Regulatory regions)
- **Negative Samples (Zeros):** 2,924 sequences (Non-enhancers)
- **Sequence Length:** Variable, ranging from 200 base pairs to 1,170 base pairs.

## Model Architecture
The deep learning pipeline is built with PyTorch and follows a hybrid **1D Convolutional Neural Network (CNN) + Bidirectional Long Short-Term Memory (BiLSTM)** framework. 

### 1. Data Encoding
Each nucleotide (A, C, G, T) is mapped to an integer and encoded using an `Embedding` layer. Unknown or padded characters ('N' or padding spaces) are explicitly handled to ensure length invariance. 
- Maximum padding is set to **1,170** base pairs.

### 2. Spatial Feature Extraction (1D CNN)
The embedding vectors are fed into a 1D Convolution layer.
- **Goal:** To act as a "motif scanner". In genomics, 1D CNNs successfully identify local transcription factor binding sites (TFBS) and conserved motifs, acting similarly to Position Weight Matrices (PWMs).
- **Settings:** Output features are extracted, pooled using MaxPooling, and down-sampled.

### 3. Long-Range Dependency Learning (BiLSTM)
The CNN features are routed into a Bidirectional LSTM network.
- **Goal:** To capture syntactical meaning and long-range interactions across the genome. Since enhancer activity can be influenced by contextual chromatin features up- and down-stream, the BiLSTM effectively processes data in both forward and reverse orders.

### 4. Classification Head
The output states of the BiLSTM are aggregated and passed through fully connected (Dense) layers with dropout (for regularization) and a final output node. 

### 5. Loss & Optimization
- **Imbalance Handling:** The dataset is heavily skewed (17k vs 2k). The model uses `BCEWithLogitsLoss` equipped with a `pos_weight` parameter (~0.16) to proportionately penalize false discoveries without aggressively over-predicting the majority class.
- **Optimizer:** Adam Optimizer is utilized for its adaptive parameter scaling.
- **Metrics:** Tracked metrics include ROC-AUC (primary metric for imbalanced data), F1 Score, Accuracy, and Precision/Recall.

## Execution
The `train_model.py` script contains the end-to-end training and evaluation loop. To execute it:

```bash
# Ensure dependencies are met
pip install torch pandas scikit-learn openpyxl

# Start the training process
python train_model.py
```

The script automatically processes the Excel file, initiates training with Stratified Folds, prints batch-wise training metrics, validates on a 20% hold-out set, and saves the trained `.pth` weights to your directory.
