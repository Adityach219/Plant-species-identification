"""
Quick start script to demonstrate the plant identification system
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.plant_classifier import PlantClassifier
from data.preprocessor import ImagePreprocessor
from utils.helpers import create_model_config, create_dataset_info


def create_demo_model():
    """Create a simple demo model for testing"""
    print("Creating demo model...")
    
    # Create a simple classifier
    classifier = PlantClassifier(num_classes=5)
    model = classifier.create_model("custom")
    
    # Save the model (untrained, just for demo)
    os.makedirs("models", exist_ok=True)
    model_path = "models/demo_model.h5"
    model.save(model_path)
    
    # Create demo class names
    class_names = ["Rose", "Sunflower", "Oak", "Maple", "Pine"]
    classes_path = "models/demo_classes.json"
    with open(classes_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"✅ Demo model saved to: {model_path}")
    print(f"✅ Demo classes saved to: {classes_path}")
    
    return model_path, classes_path


def create_sample_image():
    """Create a sample plant image for testing"""
    print("Creating sample test image...")
    
    # Create a simple colored image (green for plant-like appearance)
    image = Image.new('RGB', (300, 200), color=(34, 139, 34))  # Forest green
    
    # Add some variation to make it more interesting
    pixels = image.load()
    for i in range(image.width):
        for j in range(image.height):
            # Add some random variation
            r, g, b = pixels[i, j]
            variation = np.random.randint(-20, 20)
            g = max(0, min(255, g + variation))
            pixels[i, j] = (r, g, b)
    
    # Save sample image
    os.makedirs("data/sample", exist_ok=True)
    sample_path = "data/sample/test_plant.jpg"
    image.save(sample_path)
    
    print(f"✅ Sample image saved to: {sample_path}")
    return sample_path


def test_prediction_pipeline(model_path, classes_path, image_path):
    """Test the complete prediction pipeline"""
    print("\n🔬 Testing prediction pipeline...")
    
    # Load model
    classifier = PlantClassifier()
    classifier.load_model(model_path)
    
    # Load class names
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    classifier.class_names = class_names
    
    # Preprocess image
    preprocessor = ImagePreprocessor()
    processed_image = preprocessor.preprocess_image(image_path)
    
    # Make prediction
    predicted_class, confidence = classifier.predict(processed_image)
    print(f"🌿 Predicted species: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    # Get top 3 predictions
    top_predictions = classifier.predict_top_k(processed_image, k=3)
    print("\n🏆 Top 3 predictions:")
    for i, (species, conf) in enumerate(top_predictions, 1):
        print(f"  {i}. {species}: {conf:.2%}")
    
    return True


def show_project_structure():
    """Display the project structure"""
    print("\n📁 Project Structure:")
    print("Plant-species-identification/")
    print("├── src/")
    print("│   ├── plant_identifier/    # Core ML modules")
    print("│   ├── data/               # Data processing")
    print("│   ├── models/             # Model architectures")
    print("│   └── utils/              # Helper functions")
    print("├── webapp/                 # Web applications")
    print("├── data/                   # Dataset storage")
    print("├── models/                 # Trained models")
    print("├── tests/                  # Unit tests")
    print("├── config/                 # Configuration files")
    print("├── requirements.txt        # Python dependencies")
    print("├── setup.py               # Package setup")
    print("├── Dockerfile             # Docker configuration")
    print("└── README.md              # Documentation")


def show_usage_examples():
    """Display usage examples"""
    print("\n🚀 Usage Examples:")
    print("")
    print("1. Train a model:")
    print("   python src/plant_identifier/train.py --data_dir data/training --epochs 50")
    print("")
    print("2. Make predictions:")
    print("   python src/plant_identifier/predict.py --image test.jpg --model models/my_model.h5")
    print("")
    print("3. Start web interface:")
    print("   streamlit run webapp/app.py")
    print("")
    print("4. Start API server:")
    print("   python webapp/flask_api.py --model models/my_model.h5")
    print("")
    print("5. Run tests:")
    print("   python -m pytest tests/")
    print("")
    print("6. Build with Docker:")
    print("   docker build -t plant-identifier .")
    print("   docker run -p 8501:8501 plant-identifier")


def main():
    """Main function to run the quick start demo"""
    print("🌱 Plant Species Identification - Quick Start")
    print("=" * 50)
    
    # Show project structure
    show_project_structure()
    
    # Create configuration
    print("\n⚙️  Setting up configuration...")
    create_model_config("config/model_config.json")
    print("✅ Default configuration created")
    
    # Create demo model
    model_path, classes_path = create_demo_model()
    
    # Create sample image
    image_path = create_sample_image()
    
    # Test prediction pipeline
    try:
        test_prediction_pipeline(model_path, classes_path, image_path)
        print("\n✅ Prediction pipeline test successful!")
    except Exception as e:
        print(f"\n❌ Prediction pipeline test failed: {e}")
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Quick start demo completed!")
    print("\n📖 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add your plant images to data/training/")
    print("3. Train a real model with your data")
    print("4. Start the web interface to identify plants")
    print("\nFor detailed instructions, see the README.md file.")


if __name__ == "__main__":
    main()