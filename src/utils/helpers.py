"""
Utility functions for plant species identification
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import requests
from urllib.parse import urlparse


def download_sample_data(data_dir: str = "data/sample") -> None:
    """
    Download sample plant images for testing
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample plant image URLs (placeholder URLs - replace with actual ones)
    sample_images = {
        "rose": [
            "https://example.com/rose1.jpg",
            "https://example.com/rose2.jpg",
        ],
        "sunflower": [
            "https://example.com/sunflower1.jpg",
            "https://example.com/sunflower2.jpg",
        ],
        "oak_leaf": [
            "https://example.com/oak1.jpg",
            "https://example.com/oak2.jpg",
        ]
    }
    
    print("Note: To use this function, replace placeholder URLs with actual image URLs")
    print("For now, manually add sample images to the data/sample directory")
    
    # Create sample directories
    for species in sample_images.keys():
        species_dir = os.path.join(data_dir, species)
        os.makedirs(species_dir, exist_ok=True)
        print(f"Created directory: {species_dir}")


def create_model_config(config_path: str = "config/model_config.json") -> None:
    """
    Create default model configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config = {
        "model": {
            "input_shape": [224, 224, 3],
            "num_classes": 100,
            "architecture": "custom"
        },
        "training": {
            "batch_size": 32,
            "epochs": 50,
            "validation_split": 0.2,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        "data": {
            "target_size": [224, 224],
            "augmentation": {
                "rotation_range": 20,
                "horizontal_flip": True,
                "zoom_range": 0.1
            }
        },
        "callbacks": {
            "early_stopping": {
                "monitor": "val_accuracy",
                "patience": 10
            },
            "reduce_lr": {
                "monitor": "val_loss",
                "factor": 0.2,
                "patience": 5
            }
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model configuration saved to: {config_path}")


def visualize_training_history(history_path: str, save_path: str = None) -> None:
    """
    Visualize training history from JSON file
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def evaluate_model_performance(predictions: List[Dict], 
                             save_path: str = None) -> Dict:
    """
    Evaluate model performance from prediction results
    """
    if not predictions:
        return {}
    
    # Calculate metrics
    total_predictions = len(predictions)
    high_confidence = len([p for p in predictions if p.get('confidence', 0) >= 0.8])
    medium_confidence = len([p for p in predictions if 0.6 <= p.get('confidence', 0) < 0.8])
    low_confidence = len([p for p in predictions if p.get('confidence', 0) < 0.6])
    
    avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
    
    metrics = {
        'total_predictions': total_predictions,
        'high_confidence_count': high_confidence,
        'medium_confidence_count': medium_confidence,
        'low_confidence_count': low_confidence,
        'high_confidence_percentage': (high_confidence / total_predictions) * 100,
        'medium_confidence_percentage': (medium_confidence / total_predictions) * 100,
        'low_confidence_percentage': (low_confidence / total_predictions) * 100,
        'average_confidence': float(avg_confidence)
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Performance metrics saved to: {save_path}")
    
    return metrics


def create_dataset_info(data_dir: str, info_path: str = None) -> Dict:
    """
    Create dataset information summary
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return {}
    
    dataset_info = {
        'data_directory': data_dir,
        'classes': [],
        'total_images': 0,
        'images_per_class': {}
    }
    
    # Scan directory structure
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Count images in this class directory
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(image_extensions)]
            
            if images:  # Only include if it has images
                dataset_info['classes'].append(item)
                dataset_info['images_per_class'][item] = len(images)
                dataset_info['total_images'] += len(images)
    
    # Sort classes alphabetically
    dataset_info['classes'].sort()
    dataset_info['num_classes'] = len(dataset_info['classes'])
    
    # Calculate statistics
    if dataset_info['images_per_class']:
        image_counts = list(dataset_info['images_per_class'].values())
        dataset_info['min_images_per_class'] = min(image_counts)
        dataset_info['max_images_per_class'] = max(image_counts)
        dataset_info['avg_images_per_class'] = np.mean(image_counts)
    
    if info_path:
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"Dataset info saved to: {info_path}")
    
    return dataset_info


def print_dataset_summary(dataset_info: Dict) -> None:
    """
    Print a formatted summary of dataset information
    """
    print("Dataset Summary:")
    print("=" * 50)
    print(f"Data Directory: {dataset_info.get('data_directory', 'N/A')}")
    print(f"Number of Classes: {dataset_info.get('num_classes', 0)}")
    print(f"Total Images: {dataset_info.get('total_images', 0)}")
    
    if 'min_images_per_class' in dataset_info:
        print(f"Images per Class: {dataset_info['min_images_per_class']}-{dataset_info['max_images_per_class']} (avg: {dataset_info['avg_images_per_class']:.1f})")
    
    print("\nClasses:")
    for i, class_name in enumerate(dataset_info.get('classes', [])[:10]):
        count = dataset_info.get('images_per_class', {}).get(class_name, 0)
        print(f"  {i+1:2d}. {class_name}: {count} images")
    
    if len(dataset_info.get('classes', [])) > 10:
        print(f"  ... and {len(dataset_info['classes']) - 10} more classes")


def validate_image_dataset(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Validate image dataset and return lists of valid and invalid files
    """
    valid_files = []
    invalid_files = []
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return valid_files, invalid_files
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                try:
                    # Try to open the image to validate it
                    from PIL import Image
                    with Image.open(file_path) as img:
                        img.verify()  # Verify it's a valid image
                    valid_files.append(file_path)
                except Exception as e:
                    invalid_files.append(f"{file_path}: {str(e)}")
    
    print(f"Dataset validation complete:")
    print(f"  Valid images: {len(valid_files)}")
    print(f"  Invalid images: {len(invalid_files)}")
    
    if invalid_files:
        print("\nInvalid files:")
        for invalid in invalid_files[:5]:  # Show first 5
            print(f"  {invalid}")
        if len(invalid_files) > 5:
            print(f"  ... and {len(invalid_files) - 5} more")
    
    return valid_files, invalid_files


if __name__ == "__main__":
    # Example usage
    print("Plant Identification Utilities")
    print("Available functions:")
    print("  - download_sample_data()")
    print("  - create_model_config()")
    print("  - visualize_training_history()")
    print("  - evaluate_model_performance()")
    print("  - create_dataset_info()")
    print("  - validate_image_dataset()")
    
    # Create default configuration
    create_model_config()
    print("\nDefault model configuration created!")