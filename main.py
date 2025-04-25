import numpy as np
import matplotlib.pyplot as plt
import os
from dim import reduce_dimensions
import time


if __name__ == "__main__":
    # Create the directory to store embeddings if it doesn't exist
    os.makedirs('./emb', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    # Check if preprocessed data exists
    if not os.path.exists('./data/processed/balanced_train_images.npy') or \
       not os.path.exists('./data/processed/balanced_train_labels.npy') or \
       not os.path.exists('./data/processed/balanced_test_images.npy') or \
       not os.path.exists('./data/processed/balanced_test_labels.npy'):
        print("Preprocessed data not found. Please run preprocess.py first.")
        exit(1)

    # Load preprocessed balanced datasets
    train_images = np.load('./data/processed/balanced_train_images.npy')
    train_labels = np.load('./data/processed/balanced_train_labels.npy')
    test_images = np.load('./data/processed/balanced_test_images.npy')
    test_labels = np.load('./data/processed/balanced_test_labels.npy')

    print(f"train_images.shape: {train_images.shape}")
    print(f"train_labels.shape: {train_labels.shape}")
    print(f"test_images.shape: {test_images.shape}")
    print(f"test_labels.shape: {test_labels.shape}")

    # Verify balance of digits in training set
    print("Training set distribution:")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"Digit {digit}: {count} images")
    
    # Verify balance of digits in test set
    print("Test set distribution:")
    for digit in range(10):
        count = np.sum(test_labels == digit)
        print(f"Digit {digit}: {count} images")

    # Set n_components=2 for visualization
    n_components = 2

    # List of methods to apply
    methods = [
        'pca', 'lda', 'nmf', 'random_projection', 'metric_mds',
        'tsne', 'umap', 'isomap', 'lle', 'laplacian', 'nonmetric_mds'
    ]

    # Dictionary to store timing results
    timing_results = {}

    # Process and visualize each method separately
    for method in methods:
        print(f"Processing {method}...")
        
        # Check if embeddings already exist
        train_emb_path = f'./emb/{method}_train_2d.npy'
        test_emb_path = f'./emb/{method}_test_2d.npy'
        plot_path = f'./plots/{method}_2d_comparison.png'
        
        if os.path.exists(train_emb_path) and os.path.exists(test_emb_path) and os.path.exists(plot_path):
            print(f"Embeddings and plot for {method} already exist. Loading from files...")
            train_embeddings = np.load(train_emb_path)
            test_embeddings = np.load(test_emb_path)
            
            print(f"✓ Using existing {method} embeddings from {train_emb_path} and {test_emb_path}")
            print(f"✓ Using existing {method} visualization from {plot_path}")
            print("-" * 50)
            continue
        
        # Apply dimensionality reduction to training set with timing
        print(f"Applying {method} to training set...")
        start_time = time.time()
        
        if method == 'lda':
            train_embeddings, model = reduce_dimensions(train_images, method=method, 
                                                    n_components=n_components, 
                                                    y=train_labels)
        else:
            train_embeddings, model = reduce_dimensions(train_images, method=method, 
                                                    n_components=n_components)
        
        train_time = time.time() - start_time
        
        # Apply dimensionality reduction to test set
        print(f"Applying {method} to test set...")
        start_time = time.time()
        
        if method == 'lda':
            test_embeddings, _ = reduce_dimensions(test_images, method=method, 
                                                n_components=n_components, 
                                                y=test_labels)
        else:
            test_embeddings, _ = reduce_dimensions(test_images, method=method, 
                                                n_components=n_components)
        
        test_time = time.time() - start_time
        
        # Total execution time for timing comparison
        execution_time = train_time + test_time
        timing_results[method] = execution_time
        
        print(f"Time taken: Training: {train_time:.2f}s, Test: {test_time:.2f}s, Total: {execution_time:.2f}s")
        
        # Save embeddings
        np.save(train_emb_path, train_embeddings)
        np.save(test_emb_path, test_embeddings)
        
        # Create a 2x1 subplot figure - training vs test embeddings
        plt.figure(figsize=(16, 8))
        
        # Plot training embeddings
        plt.subplot(1, 2, 1)
        scatter_train = plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], 
                            c=train_labels, cmap='tab10', alpha=0.6, s=5)
        plt.title(f'{method.upper()} - Training Set (Time: {train_time:.2f}s)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter_train, label='Digit')
        plt.grid(alpha=0.3)
        
        # Plot test embeddings
        plt.subplot(1, 2, 2)
        scatter_test = plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], 
                           c=test_labels, cmap='tab10', alpha=0.6, s=5)
        plt.title(f'{method.upper()} - Test Set (Time: {test_time:.2f}s)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter_test, label='Digit')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plot_path, dpi=300)
        
        # Close the figure to free memory
        plt.close()
        
        print(f"✓ {method} embeddings saved to {train_emb_path} and {test_emb_path}")
        print(f"✓ {method} visualization saved to {plot_path}")
        print("-" * 50)

    # Check if timing comparison already exists
    timing_plot_path = './plots/timing_comparison.png'
    timing_results_path = './emb/timing_results.txt'
    
    if not (os.path.exists(timing_plot_path) and os.path.exists(timing_results_path) and all(timing_results.values())):
        # Create a timing comparison bar chart
        plt.figure(figsize=(14, 6))

        # Sort methods by execution time
        sorted_methods = sorted(timing_results.items(), key=lambda x: x[1])
        method_names = [method.upper() for method, _ in sorted_methods]
        execution_times = [time for _, time in sorted_methods]

        # Create bar chart with logarithmic scale for better visualization
        bars = plt.bar(method_names, execution_times, color='skyblue')
        plt.yscale('log')  # Use log scale for better visualization of time differences
        plt.xlabel('Dimensionality Reduction Method')
        plt.ylabel('Execution Time (seconds, log scale)')
        plt.title('Execution Time Comparison of Dimensionality Reduction Methods')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # Add time labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', rotation=0)

        plt.tight_layout()
        plt.savefig(timing_plot_path, dpi=300)
        plt.close()

        # Save timing results to a file
        with open(timing_results_path, 'w') as f:
            f.write("Method,Train_Time(s),Test_Time(s),Total_Time(s)\n")
            for method in method_names:
                method_lower = method.lower()
                train_time = timing_results.get(method_lower, 0) / 2  # Approximate split for this example
                test_time = timing_results.get(method_lower, 0) / 2   # Approximate split for this example
                total_time = timing_results.get(method_lower, 0)
                f.write(f"{method},{train_time:.4f},{test_time:.4f},{total_time:.4f}\n")
        
        print("✓ Timing comparison saved to timing_plot_path")
    else:
        print("✓ Using existing timing comparison files")

    print("Completed analysis of all dimensionality reduction methods!")