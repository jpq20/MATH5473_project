import matplotlib.pyplot as plt
import numpy as np
import os

def create_separate_combined_plots():
    # List all dimensionality reduction methods
    methods = [
        'pca', 'lda', 'nmf', 'random_projection', 'metric_mds',
        'tsne', 'umap', 'isomap', 'lle', 'laplacian', 'nonmetric_mds'
    ]
    
    # Load labels
    train_labels = np.load('./data/processed/balanced_train_labels.npy')
    test_labels = np.load('./data/processed/balanced_test_labels.npy')
    
    # Create separate plots for training and testing
    for data_type in ['train', 'test']:
        # Create figure
        fig, axes = plt.subplots(4, 3, figsize=(18, 22))
        fig.suptitle(f'{data_type.title()} Set Embeddings Across Different Dimensionality Reduction Methods', 
                     fontsize=20, y=0.98)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Create plot for each method
        for i, method in enumerate(methods):
            emb_path = f'./emb/{method}_{data_type}_2d.npy'
            
            if os.path.exists(emb_path):
                # Load embeddings
                embeddings = np.load(emb_path)
                
                # Use appropriate labels
                labels = train_labels if data_type == 'train' else test_labels
                
                # Plot
                scatter = axes[i].scatter(
                    embeddings[:, 0], embeddings[:, 1],
                    c=labels, cmap='tab10', alpha=0.6, s=5
                )
                
                axes[i].set_title(f'{method.upper()}', fontsize=14)
                axes[i].set_xlabel('Component 1')
                axes[i].set_ylabel('Component 2')
                axes[i].grid(alpha=0.3)
                
                # Add colorbar for the first plot only
                if i == 0:
                    cbar = plt.colorbar(scatter, ax=axes[i])
                    cbar.set_label('Digit')
            else:
                axes[i].text(0.5, 0.5, f"No {method} embeddings found", 
                            ha='center', va='center',
                            transform=axes[i].transAxes)
        
        # Add a small legend to each plot
        for i in range(len(methods), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        plt.savefig(f'./plots/combined_{data_type}_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined {data_type} embeddings saved to ./plots/combined_{data_type}_embeddings.png")

if __name__ == "__main__":
    create_separate_combined_plots()