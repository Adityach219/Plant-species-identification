"""
Prediction script for plant species identification
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.plant_classifier import PlantClassifier
from src.data.preprocessor import ImagePreprocessor


def load_class_names(classes_path: str) -> list:
    """Load class names from JSON file"""
    with open(classes_path, 'r') as f:
        return json.load(f)


def predict_image(image_path: str, 
                 model_path: str, 
                 classes_path: str = None,
                 top_k: int = 5) -> None:
    """
    Predict plant species from image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        classes_path: Path to class names JSON file
        top_k: Number of top predictions to show
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    classifier = PlantClassifier()
    classifier.load_model(model_path)
    
    # Load class names if provided
    if classes_path and os.path.exists(classes_path):
        classifier.class_names = load_class_names(classes_path)
        print(f"Loaded {len(classifier.class_names)} class names")
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    preprocessor = ImagePreprocessor()
    
    try:
        # Try loading as file path first
        image = preprocessor.preprocess_image(image_path)
    except:
        # If that fails, try loading as PIL image
        pil_image = Image.open(image_path)
        image = preprocessor.preprocess_pil_image(pil_image)
    
    # Make prediction
    print(f"Making prediction...")
    
    if top_k == 1:
        predicted_class, confidence = classifier.predict(image)
        print(f"\nPrediction:")
        print(f"  Species: {predicted_class}")
        print(f"  Confidence: {confidence:.2%}")
    else:
        predictions = classifier.predict_top_k(image, k=top_k)
        print(f"\nTop {len(predictions)} predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {confidence:.2%}")


def predict_batch(images_dir: str,
                 model_path: str,
                 classes_path: str = None,
                 output_file: str = None) -> None:
    """
    Predict plant species for batch of images
    
    Args:
        images_dir: Directory containing images
        model_path: Path to trained model
        classes_path: Path to class names JSON file
        output_file: Optional file to save results
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    classifier = PlantClassifier()
    classifier.load_model(model_path)
    
    # Load class names if provided
    if classes_path and os.path.exists(classes_path):
        classifier.class_names = load_class_names(classes_path)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    preprocessor = ImagePreprocessor()
    results = []
    
    for i, filename in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {filename}")
        
        image_path = os.path.join(images_dir, filename)
        try:
            image = preprocessor.preprocess_image(image_path)
            predicted_class, confidence = classifier.predict(image)
            
            result = {
                'filename': filename,
                'predicted_species': predicted_class,
                'confidence': float(confidence)
            }
            results.append(result)
            
            print(f"  -> {predicted_class} ({confidence:.2%})")
            
        except Exception as e:
            print(f"  -> Error: {str(e)}")
            results.append({
                'filename': filename,
                'error': str(e)
            })
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    successful = len([r for r in results if 'predicted_species' in r])
    print(f"\nBatch processing completed:")
    print(f"  Successfully processed: {successful}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description='Predict plant species from images')
    parser.add_argument('--image', type=str,
                       help='Path to single image file')
    parser.add_argument('--batch', type=str,
                       help='Directory containing multiple images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--classes', type=str,
                       help='Path to class names JSON file')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show (for single image)')
    parser.add_argument('--output', type=str,
                       help='Output file for batch results (JSON format)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        print("Error: Either --image or --batch must be specified")
        sys.exit(1)
    
    if args.image and args.batch:
        print("Error: Cannot specify both --image and --batch")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        sys.exit(1)
    
    if args.classes and not os.path.exists(args.classes):
        print(f"Warning: Classes file '{args.classes}' does not exist")
    
    # Run prediction
    try:
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file '{args.image}' does not exist")
                sys.exit(1)
            
            predict_image(
                image_path=args.image,
                model_path=args.model,
                classes_path=args.classes,
                top_k=args.top_k
            )
        
        elif args.batch:
            if not os.path.exists(args.batch):
                print(f"Error: Images directory '{args.batch}' does not exist")
                sys.exit(1)
            
            predict_batch(
                images_dir=args.batch,
                model_path=args.model,
                classes_path=args.classes,
                output_file=args.output
            )
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()