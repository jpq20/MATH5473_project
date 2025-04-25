import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import pandas as pd
import seaborn as sns


def load_data(method, dimension='2d'):
    """
    Load the embeddings and labels
    
    Parameters:
    -----------
    method : str
        Dimensionality reduction method name
    dimension : str, optional (default='2d')
        Dimension of the embeddings ('2d' or '50d')
    
    Returns:
    --------
    X : array-like
        Embeddings
    y : array-like
        Labels
    """
    # Load embeddings
    embedding_path = f'./emb/{method}_train_{dimension}.npy'
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    X = np.load(embedding_path)
    
    # Load labels
    labels_path = './data/processed/balanced_train_labels.npy'
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}. Run preprocess.py first.")
    
    y = np.load(labels_path)
    
    # Verify sizes match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch between embeddings ({X.shape[0]} samples) and labels ({y.shape[0]} samples)")
    
    return X, y

def evaluate_classifier(X, y, classifier_name='knn', test_size=0.2, random_state=42):
    """
    Evaluate a classifier on the embeddings
    
    Parameters:
    -----------
    X : array-like
        Embeddings
    y : array-like
        Labels
    classifier_name : str, optional (default='knn')
        Classifier to use ('knn', 'svm', or 'rf')
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    accuracy : float
        Classification accuracy
    train_time : float
        Training time in seconds
    test_time : float
        Testing time in seconds
    model : object
        Trained classifier
    y_pred : array-like
        Predicted labels
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize the classifier
    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'svm':
        classifier = SVC(gamma='scale', random_state=random_state)
    elif classifier_name == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    # Train the classifier
    train_start = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Test the classifier
    test_start = time.time()
    y_pred = classifier.predict(X_test)
    test_time = time.time() - test_start
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, train_time, test_time, classifier, y_pred, y_test

def run_classification_benchmark():
    """
    Run classification benchmark on different embeddings and classifiers
    """
    # Create directories
    os.makedirs('./class', exist_ok=True)
    
    # Methods to evaluate
    methods = [
        'pca', 'lda', 'nmf', 'random_projection', 'metric_mds',
        'tsne', 'umap', 'isomap', 'lle', 'laplacian', 'nonmetric_mds'
    ]
    
    # Classifiers to evaluate
    classifiers = ['knn', 'svm', 'rf']
    
    # Results dictionaries
    results = []
    
    # Evaluate each method with each classifier
    for method in methods:
        print(f"Evaluating {method}...")
        
        try:
            # Load the data
            X, y = load_data(method, dimension='2d')
            print(f"  Loaded embeddings with shape {X.shape} and {len(y)} labels")
            
            for classifier_name in classifiers:
                print(f"  Using {classifier_name}...")
                
                try:
                    # Evaluate the classifier
                    accuracy, train_time, test_time, model, y_pred, y_test = evaluate_classifier(
                        X, y, classifier_name=classifier_name)
                    
                    # Add results to the dictionary
                    results.append({
                        'method': method,
                        'classifier': classifier_name,
                        'accuracy': accuracy,
                        'train_time': train_time,
                        'test_time': test_time
                    })
                    
                    print(f"    Accuracy: {accuracy:.4f}, Train time: {train_time:.4f}s, Test time: {test_time:.4f}s")
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix: {method.upper()} + {classifier_name.upper()}')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.savefig(f'./class/{method}_{classifier_name}_confusion.png', dpi=300)
                    plt.close()
                    
                except Exception as e:
                    print(f"    Error with {classifier_name}: {e}")
                    results.append({
                        'method': method,
                        'classifier': classifier_name,
                        'accuracy': None,
                        'train_time': None,
                        'test_time': None
                    })
        
        except Exception as e:
            print(f"Error with {method}: {e}")
            for classifier_name in classifiers:
                results.append({
                    'method': method,
                    'classifier': classifier_name,
                    'accuracy': None,
                    'train_time': None,
                    'test_time': None
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('./class/classification_results.csv', index=False)
    
    # Create accuracy comparison plot
    plt.figure(figsize=(14, 8))
    
    # Reshape data for seaborn
    pivot_df = results_df.pivot(index='method', columns='classifier', values='accuracy')
    
    # Plot heatmap
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f', cbar_kws={'label': 'Accuracy'})
    plt.title('Classification Accuracy by Method and Classifier')
    plt.xlabel('Classifier')
    plt.ylabel('Dimensionality Reduction Method')
    plt.tight_layout()
    plt.savefig('./class/accuracy_comparison.png', dpi=300)
    plt.close()
    
    # Create bar chart for each classifier
    for classifier_name in classifiers:
        classifier_df = results_df[results_df['classifier'] == classifier_name].sort_values('accuracy', ascending=False)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classifier_df['method'].str.upper(), classifier_df['accuracy'], color='skyblue')
        plt.xlabel('Dimensionality Reduction Method')
        plt.ylabel('Classification Accuracy')
        plt.title(f'Classification Accuracy with {classifier_name.upper()}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add accuracy labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'./class/{classifier_name}_accuracy.png', dpi=300)
        plt.close()
    
    # Create a combined plot for all classifiers
    plt.figure(figsize=(15, 8))
    
    # Get unique methods and sort alphabetically
    methods = sorted(results_df['method'].unique())
    
    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(methods))
    
    # Filter out NaN values for each classifier
    knn_df = results_df[results_df['classifier'] == 'knn'].set_index('method')
    svm_df = results_df[results_df['classifier'] == 'svm'].set_index('method')
    rf_df = results_df[results_df['classifier'] == 'rf'].set_index('method')
    
    # Get accuracies (replace NaN with 0)
    knn_acc = [knn_df.loc[m, 'accuracy'] if m in knn_df.index and not pd.isna(knn_df.loc[m, 'accuracy']) else 0 for m in methods]
    svm_acc = [svm_df.loc[m, 'accuracy'] if m in svm_df.index and not pd.isna(svm_df.loc[m, 'accuracy']) else 0 for m in methods]
    rf_acc = [rf_df.loc[m, 'accuracy'] if m in rf_df.index and not pd.isna(rf_df.loc[m, 'accuracy']) else 0 for m in methods]
    
    # Plot bars
    plt.bar(index - bar_width, knn_acc, bar_width, label='KNN', color='skyblue')
    plt.bar(index, svm_acc, bar_width, label='SVM', color='lightgreen')
    plt.bar(index + bar_width, rf_acc, bar_width, label='RF', color='salmon')
    
    # Add labels and title
    plt.xlabel('Dimensionality Reduction Method')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy Comparison Across Methods and Classifiers')
    plt.xticks(index, [m.upper() for m in methods], rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('./class/combined_accuracy_comparison.png', dpi=300)
    plt.close()
    
    print("Classification benchmark completed!")
    print(f"Results saved to ./class/classification_results.csv")
    print(f"Plots saved to ./class/ directory")

def evaluate_specific_method(method, classifier_name='knn'):
    """
    Detailed evaluation of a specific method with a specific classifier
    
    Parameters:
    -----------
    method : str
        Dimensionality reduction method name
    classifier_name : str, optional (default='knn')
        Classifier to use ('knn', 'svm', or 'rf')
    """
    # Create output directory
    os.makedirs('./class', exist_ok=True)
    
    # Load the data
    X, y = load_data(method, dimension='2d')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize the classifier
    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'svm':
        classifier = SVC(gamma='scale', random_state=42)
    elif classifier_name == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Test the classifier
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Method: {method.upper()}, Classifier: {classifier_name.upper()}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save the classification report to a text file
    with open(f'./class/{method}_{classifier_name}_report.txt', 'w') as f:
        f.write(f"Method: {method.upper()}, Classifier: {classifier_name.upper()}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {method.upper()} + {classifier_name.upper()}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'./class/{method}_{classifier_name}_detailed_confusion.png', dpi=300)
    plt.close()
    
    # Visualize the embeddings with misclassifications highlighted
    X, y = load_data(method, dimension='2d')
    plt.figure(figsize=(12, 10))
    
    # Predict on all data
    y_all_pred = classifier.predict(X)
    
    # Create a mask for correct and incorrect predictions
    correct = y_all_pred == y
    incorrect = ~correct
    
    # Plot correct predictions
    plt.scatter(X[correct, 0], X[correct, 1], c=y[correct], cmap='tab10', 
                alpha=0.6, s=5, label='Correct')
    
    # Plot incorrect predictions
    plt.scatter(X[incorrect, 0], X[incorrect, 1], marker='x', c='red', 
                s=20, label='Misclassified')
    
    plt.title(f'{method.upper()} - 2D Embedding with {classifier_name.upper()} Predictions')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Digit Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'./class/{method}_{classifier_name}_misclassification.png', dpi=300)
    plt.close()
    
    return accuracy, classifier

if __name__ == "__main__":
    # Check if preprocessed data exists
    if not os.path.exists('./data/processed/balanced_train_labels.npy'):
        print("Preprocessed data not found. Please run preprocess.py first.")
    else:
        # Create the directory for results
        os.makedirs('./class', exist_ok=True)
        
        print("Running classification benchmark...")
        run_classification_benchmark()