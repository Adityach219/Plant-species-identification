"""
Plant Species Identification Model
Implements CNN architecture for classifying plant species from images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List


class PlantClassifier:
    """
    CNN-based plant species classifier
    """
    
    def __init__(self, num_classes: int = 100, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize the plant classifier
        
        Args:
            num_classes: Number of plant species to classify
            input_shape: Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.class_names = []
        
    def create_model(self, model_type: str = "custom") -> keras.Model:
        """
        Create CNN model for plant classification
        
        Args:
            model_type: Type of model architecture ("custom", "resnet", "efficientnet")
            
        Returns:
            Compiled Keras model
        """
        if model_type == "custom":
            model = self._create_custom_model()
        elif model_type == "resnet":
            model = self._create_resnet_model()
        elif model_type == "efficientnet":
            model = self._create_efficientnet_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model = model
        return model
    
    def _create_custom_model(self) -> keras.Model:
        """Create custom CNN architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(2),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_resnet_model(self) -> keras.Model:
        """Create ResNet-based transfer learning model"""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_efficientnet_model(self) -> keras.Model:
        """Create EfficientNet-based transfer learning model"""
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model"""
        self.model = keras.models.load_model(model_path)
        
    def save_model(self, model_path: str) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Create or train a model first.")
        self.model.save(model_path)
        
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict plant species from image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or create a model first.")
            
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        predictions = self.model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        if self.class_names:
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
            
        return predicted_class, float(confidence)
    
    def predict_top_k(self, image: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k predictions
        
        Args:
            image: Preprocessed image array
            k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or create a model first.")
            
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        predictions = self.model.predict(image)
        top_k_indices = np.argsort(predictions[0])[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if self.class_names:
                class_name = self.class_names[idx]
            else:
                class_name = f"Class_{idx}"
            confidence = float(predictions[0][idx])
            results.append((class_name, confidence))
            
        return results