"""
Test suite for plant species identification system
"""

import unittest
import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.plant_classifier import PlantClassifier
from data.preprocessor import ImagePreprocessor, DataLoader


class TestPlantClassifier(unittest.TestCase):
    """Test cases for PlantClassifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = PlantClassifier(num_classes=5, input_shape=(224, 224, 3))
    
    def test_model_creation(self):
        """Test model creation"""
        model = self.classifier.create_model("custom")
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 5))
    
    def test_model_types(self):
        """Test different model architectures"""
        for model_type in ["custom", "resnet", "efficientnet"]:
            model = self.classifier.create_model(model_type)
            self.assertIsNotNone(model)
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error"""
        with self.assertRaises(ValueError):
            self.classifier.create_model("invalid_type")
    
    def test_prediction_without_model(self):
        """Test prediction fails without loaded model"""
        dummy_image = np.random.rand(224, 224, 3)
        with self.assertRaises(ValueError):
            self.classifier.predict(dummy_image)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Create a temporary test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Create a simple test image
        test_image = Image.new('RGB', (300, 200), color='green')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_preprocess_image(self):
        """Test image preprocessing from file"""
        processed = self.preprocessor.preprocess_image(self.test_image_path)
        
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertTrue(processed.min() >= 0.0)
        self.assertTrue(processed.max() <= 1.0)
    
    def test_preprocess_pil_image(self):
        """Test PIL image preprocessing"""
        pil_image = Image.new('RGB', (300, 200), color='blue')
        processed = self.preprocessor.preprocess_pil_image(pil_image)
        
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertTrue(processed.min() >= 0.0)
        self.assertTrue(processed.max() <= 1.0)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing"""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            path = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            image = Image.new('RGB', (100, 100), color='red')
            image.save(path)
            image_paths.append(path)
        
        batch = self.preprocessor.preprocess_batch(image_paths)
        
        self.assertEqual(batch.shape, (3, 224, 224, 3))
        self.assertTrue(batch.min() >= 0.0)
        self.assertTrue(batch.max() <= 1.0)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_image("nonexistent.jpg")


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary dataset structure
        self.temp_dir = tempfile.mkdtemp()
        
        # Create class directories with test images
        for class_name in ['class1', 'class2']:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir)
            
            # Create a few test images in each class
            for i in range(3):
                image_path = os.path.join(class_dir, f"image_{i}.jpg")
                image = Image.new('RGB', (100, 100), color='green')
                image.save(image_path)
        
        self.data_loader = DataLoader(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_class_names(self):
        """Test getting class names"""
        class_names = self.data_loader.get_class_names()
        self.assertEqual(len(class_names), 2)
        self.assertIn('class1', class_names)
        self.assertIn('class2', class_names)
    
    def test_get_sample_images(self):
        """Test getting sample images"""
        samples = self.data_loader.get_sample_images()
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 2)  # Should be <= number of classes


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow without actual training"""
        # Create classifier
        classifier = PlantClassifier(num_classes=3)
        model = classifier.create_model("custom")
        
        # Create dummy trained model weights
        dummy_input = np.random.rand(1, 224, 224, 3)
        dummy_output = model.predict(dummy_input)
        
        # Test that model produces output of correct shape
        self.assertEqual(dummy_output.shape, (1, 3))
        
        # Test preprocessing
        preprocessor = ImagePreprocessor()
        
        # Create test image
        test_image = Image.new('RGB', (300, 200), color='red')
        processed = preprocessor.preprocess_pil_image(test_image)
        
        # Test prediction pipeline (will be random without training)
        prediction = model.predict(np.expand_dims(processed, axis=0))
        self.assertEqual(prediction.shape, (1, 3))


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPlantClassifier))
    test_suite.addTest(unittest.makeSuite(TestImagePreprocessor))
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)