import numpy as np
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.random_projection import GaussianRandomProjection
import umap
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import os

# ---------------------- Linear Methods ----------------------

def apply_pca(X, n_components=2):
    """
    Principal Component Analysis
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    pca : PCA object
        Fitted PCA model
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def apply_lda(X, y, n_components=2):
    """
    Linear Discriminant Analysis
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    lda : LinearDiscriminantAnalysis object
        Fitted LDA model
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_reduced = lda.fit_transform(X, y)
    return X_reduced, lda

def apply_nmf(X, n_components=2):
    """
    Non-negative Matrix Factorization
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Non-negative training data
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    nmf : NMF object
        Fitted NMF model
    """
    # Ensure data is non-negative
    if np.any(X < 0):
        raise ValueError("NMF requires non-negative data")
    
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    X_reduced = nmf.fit_transform(X)
    return X_reduced, nmf

def apply_random_projection(X, n_components=2):
    """
    Random Projection
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    rp : GaussianRandomProjection object
        Fitted Random Projection model
    """
    rp = GaussianRandomProjection(n_components=n_components, random_state=0)
    X_reduced = rp.fit_transform(X)
    return X_reduced, rp

def apply_metric_mds(X, n_components=2):
    """
    Metric Multidimensional Scaling
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    mds : MDS object
        Fitted MDS model
    """
    mds = MDS(n_components=n_components, metric=True, random_state=0, n_jobs=-1)
    X_reduced = mds.fit_transform(X)
    return X_reduced, mds

# ---------------------- Nonlinear Methods ----------------------

def apply_tsne(X, n_components=2, perplexity=30.0):
    """
    t-Distributed Stochastic Neighbor Embedding
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    perplexity : float, optional (default=30.0)
        Related to the number of nearest neighbors
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    tsne : TSNE object
        Fitted t-SNE model
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0)
    X_reduced = tsne.fit_transform(X)
    return X_reduced, tsne

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Uniform Manifold Approximation and Projection
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    n_neighbors : int, optional (default=15)
        Number of neighbors to consider
    min_dist : float, optional (default=0.1)
        Minimum distance between points in the embedding
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    reducer : UMAP object
        Fitted UMAP model
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=0)
    X_reduced = reducer.fit_transform(X)
    return X_reduced, reducer


def apply_isomap(X, n_components=2, n_neighbors=5):
    """
    Isomap Embedding
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    n_neighbors : int, optional (default=5)
        Number of neighbors for each sample
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    isomap : Isomap object
        Fitted Isomap model
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    X_reduced = isomap.fit_transform(X)
    return X_reduced, isomap

def apply_lle(X, n_components=2, n_neighbors=5):
    """
    Locally Linear Embedding
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    n_neighbors : int, optional (default=5)
        Number of neighbors for each sample
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    lle : LocallyLinearEmbedding object
        Fitted LLE model
    """
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, 
                                 method='standard', random_state=0)
    X_reduced = lle.fit_transform(X)
    return X_reduced, lle

def apply_laplacian_eigenmaps(X, n_components=2, n_neighbors=5):
    """
    Laplacian Eigenmaps (Spectral Embedding)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    n_neighbors : int, optional (default=5)
        Number of neighbors for each sample
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    laplacian : SpectralEmbedding object
        Fitted SpectralEmbedding model
    """
    laplacian = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, 
                                  random_state=0)
    X_reduced = laplacian.fit_transform(X)
    return X_reduced, laplacian

def apply_nonmetric_mds(X, n_components=2):
    """
    Non-metric Multidimensional Scaling
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int, optional (default=2)
        Number of components to keep
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    mds : MDS object
        Fitted MDS model
    """
    mds = MDS(n_components=n_components, metric=False, random_state=0)
    X_reduced = mds.fit_transform(X)
    return X_reduced, mds

# Example usage function
def reduce_dimensions(X, method='pca', n_components=2, **kwargs):
    """
    Apply dimensionality reduction to MNIST data
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        MNIST data (flattened images)
    method : string, optional (default='pca')
        Dimensionality reduction method to use
    n_components : int, optional (default=2)
        Number of components to keep
    **kwargs : dict
        Additional parameters for the specific method
    
    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Reduced data
    model : object
        Fitted model
    """
    methods = {
        'pca': apply_pca,
        'lda': apply_lda,
        'nmf': apply_nmf,
        'random_projection': apply_random_projection,
        'metric_mds': apply_metric_mds,
        'tsne': apply_tsne,
        'umap': apply_umap,
        'isomap': apply_isomap,
        'lle': apply_lle,
        'laplacian': apply_laplacian_eigenmaps,
        'nonmetric_mds': apply_nonmetric_mds
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(methods.keys())}")
    
    # Special case for LDA which requires labels
    if method == 'lda' and 'y' not in kwargs:
        raise ValueError("LDA requires labels (y) to be provided in kwargs")
    
    # Handle method-specific parameters
    if method == 'lda':
        return methods[method](X, kwargs['y'], n_components=n_components)
    else:
        # Update kwargs with n_components
        kwargs['n_components'] = n_components
        return methods[method](X, **kwargs)


if __name__ == "__main__":
    # Create the directory to store embeddings if it doesn't exist
    os.makedirs('./emb', exist_ok=True)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Get the data and labels
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    train_images = train_images.numpy()
    train_labels = train_labels.numpy()
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()

    # Define the number of components for reduction
    n_components = 50

    # List of methods to apply
    methods = [
        'pca', 'lda', 'nmf', 'random_projection', 'metric_mds',
        'tsne', 'umap', 'isomap', 'lle', 'laplacian', 'nonmetric_mds'
    ]

    # Apply each method and save the embeddings
    for method in methods:
        print(f"Applying {method}...")
        if method == 'lda':
            train_embeddings, _ = reduce_dimensions(train_images, method=method, 
                                                   n_components=n_components, 
                                                   y=train_labels,)
            test_embeddings, _ = reduce_dimensions(test_images, method=method, 
                                                 n_components=n_components, 
                                                 y=test_labels)
        else:
            train_embeddings, _ = reduce_dimensions(train_images, method=method, 
                                                   n_components=n_components,)
            test_embeddings, _ = reduce_dimensions(test_images, method=method, 
                                                 n_components=n_components,)
        
        # Save the embeddings
        np.save(f'./emb/{method}_train_embeddings.npy', train_embeddings)
        np.save(f'./emb/{method}_test_embeddings.npy', test_embeddings)

    print("All embeddings have been saved in the './emb' directory.")