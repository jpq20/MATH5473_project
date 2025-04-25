import numpy as np
import os
import torch
from torchvision import datasets, transforms

def preprocess_mnist():
    """
    Prepare and save a balanced subset of MNIST data and their labels.
    Creates 800 training samples and 200 test samples per digit (0-9).
    """
    print("Starting MNIST preprocessing...")
    
    # Create data directories
    os.makedirs('./data/processed', exist_ok=True)
    
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

    print(f"Original train_images.shape: {train_images.shape}")
    print(f"Original train_labels.shape: {train_labels.shape}")
    print(f"Original test_images.shape: {test_images.shape}")
    print(f"Original test_labels.shape: {test_labels.shape}")

    # Create balanced training set with 800 images per digit from the original training set
    balanced_train_images = []
    balanced_train_labels = []
    
    # Create balanced test set with 200 images per digit from the original test set
    balanced_test_images = []
    balanced_test_labels = []
    
    # Number of images per digit
    train_per_digit = 800
    test_per_digit = 200
    
    # Set random seed for reproducibility
    np.random.seed(42)

    for digit in range(10):
        # Get indices of this digit in train set
        train_digit_indices = np.where(train_labels == digit)[0]
        # Randomly sample train_per_digit images from training set
        if len(train_digit_indices) >= train_per_digit:
            train_selected_indices = np.random.choice(train_digit_indices, train_per_digit, replace=False)
        else:
            # In case there aren't enough examples of a digit
            print(f"Warning: Only {len(train_digit_indices)} examples of digit {digit} in training set")
            train_selected_indices = np.random.choice(train_digit_indices, train_per_digit, replace=True)
        
        # Add selected train images and labels
        balanced_train_images.append(train_images[train_selected_indices])
        balanced_train_labels.append(train_labels[train_selected_indices])
        
        # Get indices of this digit in test set
        test_digit_indices = np.where(test_labels == digit)[0]
        # Randomly sample test_per_digit images from test set
        if len(test_digit_indices) >= test_per_digit:
            test_selected_indices = np.random.choice(test_digit_indices, test_per_digit, replace=False)
        else:
            # In case there aren't enough examples of a digit
            print(f"Warning: Only {len(test_digit_indices)} examples of digit {digit} in test set")
            test_selected_indices = np.random.choice(test_digit_indices, test_per_digit, replace=True)
        
        # Add selected test images and labels
        balanced_test_images.append(test_images[test_selected_indices])
        balanced_test_labels.append(test_labels[test_selected_indices])

    # Combine all selected training images and labels
    balanced_train_images = np.vstack(balanced_train_images)
    balanced_train_labels = np.concatenate(balanced_train_labels)
    
    # Combine all selected test images and labels
    balanced_test_images = np.vstack(balanced_test_images)
    balanced_test_labels = np.concatenate(balanced_test_labels)
    
    # Shuffle the training data
    train_shuffle_indices = np.random.permutation(len(balanced_train_images))
    balanced_train_images = balanced_train_images[train_shuffle_indices]
    balanced_train_labels = balanced_train_labels[train_shuffle_indices]
    
    # Shuffle the test data
    test_shuffle_indices = np.random.permutation(len(balanced_test_images))
    balanced_test_images = balanced_test_images[test_shuffle_indices]
    balanced_test_labels = balanced_test_labels[test_shuffle_indices]
    
    print(f"Balanced train images shape: {balanced_train_images.shape}")
    print(f"Balanced train labels shape: {balanced_train_labels.shape}")
    print(f"Balanced test images shape: {balanced_test_images.shape}")
    print(f"Balanced test labels shape: {balanced_test_labels.shape}")
    
    # Verify balance of digits in training set
    print("Training set distribution:")
    for digit in range(10):
        count = np.sum(balanced_train_labels == digit)
        print(f"Digit {digit}: {count} images")
    
    # Verify balance of digits in test set
    print("Test set distribution:")
    for digit in range(10):
        count = np.sum(balanced_test_labels == digit)
        print(f"Digit {digit}: {count} images")
    
    # Save the balanced datasets
    np.save('./data/processed/balanced_train_images.npy', balanced_train_images)
    np.save('./data/processed/balanced_train_labels.npy', balanced_train_labels)
    np.save('./data/processed/balanced_test_images.npy', balanced_test_images)
    np.save('./data/processed/balanced_test_labels.npy', balanced_test_labels)
    
    print("✓ Balanced datasets saved to ./data/processed/")
    print("  - balanced_train_images.npy")
    print("  - balanced_train_labels.npy")
    print("  - balanced_test_images.npy")
    print("  - balanced_test_labels.npy")
    
    return {
        'train_images': balanced_train_images,
        'train_labels': balanced_train_labels,
        'test_images': balanced_test_images,
        'test_labels': balanced_test_labels
    }

if __name__ == "__main__":
    preprocess_mnist() 