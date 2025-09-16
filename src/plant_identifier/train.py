"""
Training script for plant species identification model
"""

import argparse
import os
import sys
import json
import tensorflow as tf
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.plant_classifier import PlantClassifier
from src.data.preprocessor import DataLoader


def train_model(data_dir: str, 
                model_type: str = "custom",
                epochs: int = 50,
                batch_size: int = 32,
                validation_split: float = 0.2,
                output_dir: str = "models"):
    """
    Train plant species identification model
    
    Args:
        data_dir: Directory containing training data
        model_type: Type of model architecture
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        output_dir: Directory to save trained model
    """
    print(f"Starting training with parameters:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model type: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    data_loader = DataLoader(data_dir, target_size=(224, 224))
    train_ds, val_ds = data_loader.load_dataset(
        validation_split=validation_split,
        batch_size=batch_size
    )
    
    class_names = data_loader.get_class_names()
    num_classes = len(class_names)
    print(f"Found {num_classes} plant species classes:")
    for i, class_name in enumerate(class_names[:5]):  # Show first 5
        print(f"  {i}: {class_name}")
    if len(class_names) > 5:
        print(f"  ... and {len(class_names) - 5} more")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    classifier = PlantClassifier(num_classes=num_classes)
    model = classifier.create_model(model_type=model_type)
    
    print(f"Model architecture:")
    model.summary()
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"plant_classifier_{model_type}_{timestamp}"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save class names
    class_names_path = os.path.join(output_dir, f"{model_name}_classes.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to: {class_names_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\nTraining completed!")
    print(f"Final metrics:")
    print(f"  Training Loss: {final_loss:.4f}")
    print(f"  Training Accuracy: {final_accuracy:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {final_val_accuracy:.4f}")
    
    return final_model_path, class_names_path


def main():
    parser = argparse.ArgumentParser(description='Train plant species identification model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data organized by species')
    parser.add_argument('--model_type', type=str, choices=['custom', 'resnet', 'efficientnet'],
                       default='custom', help='Type of model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        sys.exit(1)
    
    if args.validation_split <= 0 or args.validation_split >= 1:
        print("Error: validation_split must be between 0 and 1")
        sys.exit(1)
    
    # Train model
    try:
        model_path, classes_path = train_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            output_dir=args.output_dir
        )
        print(f"\nTraining successful!")
        print(f"Model saved to: {model_path}")
        print(f"Classes saved to: {classes_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()