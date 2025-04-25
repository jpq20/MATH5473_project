# Dimensionality Reduction and Classification of MNIST

This project evaluates various dimensionality reduction techniques on the MNIST dataset using different classifiers to compare performance.

## Overview

We implemented 11 dimensionality reduction methods and evaluated their effectiveness using 3 classification algorithms. The project analyzes both the visual quality of the dimension-reduced embeddings and the classification accuracy for each method.

## Structure

- `preprocess.py`: Creates balanced subsets of MNIST data and saves them to `./data/processed/`
- `dim.py`: Contains implementations of all dimensionality reduction methods
- `main.py`: Applies dimensionality reduction to data and creates visualizations
- `class.py`: Implements classification benchmarking and evaluates performance

## Dimensionality Reduction Methods

1. PCA (Principal Component Analysis)
2. LDA (Linear Discriminant Analysis)
3. NMF (Non-negative Matrix Factorization)
4. Random Projection
5. Metric MDS (Multidimensional Scaling)
6. t-SNE (t-Distributed Stochastic Neighbor Embedding)
7. UMAP (Uniform Manifold Approximation and Projection)
8. Isomap
9. LLE (Locally Linear Embedding)
10. Laplacian Eigenmaps
11. Non-metric MDS

## Classifiers

1. KNN (K-Nearest Neighbors)
2. SVM (Support Vector Machine)
3. RF (Random Forest)

## Results

The results are stored in:
- `./emb/`: Contains saved embeddings for each method
- `./plots/`: Contains visualizations of embeddings and performance comparisons
- `./class/`: Contains classification results and analysis