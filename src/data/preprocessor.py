"""
Data preprocessing utilities for plant images
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional
import tensorflow as tf


class ImagePreprocessor:
    """
    Handles image preprocessing for plant identification
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess single image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_pil_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image for prediction
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Preprocessed image array
        """
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # Resize image
        pil_image = pil_image.resize(self.target_size)
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch of preprocessed images
        """
        images = []
        for path in image_paths:
            image = self.preprocess_image(path)
            images.append(image)
            
        return np.array(images)
    
    def apply_augmentation(self, image: np.ndarray, 
                          rotation_range: float = 20.0,
                          zoom_range: float = 0.1,
                          horizontal_flip: bool = True) -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image array
            rotation_range: Range of random rotation in degrees
            zoom_range: Range of random zoom
            horizontal_flip: Whether to apply random horizontal flip
            
        Returns:
            Augmented image
        """
        # Convert to tensor for TensorFlow operations
        image_tensor = tf.convert_to_tensor(image)
        
        # Random rotation
        if rotation_range > 0:
            angle = tf.random.uniform([], -rotation_range, rotation_range)
            image_tensor = tf.keras.preprocessing.image.apply_transform(
                image_tensor, 
                transform_matrix=tf.keras.preprocessing.image.random_rotation(angle)
            )
        
        # Random horizontal flip
        if horizontal_flip:
            image_tensor = tf.image.random_flip_left_right(image_tensor)
        
        # Random zoom
        if zoom_range > 0:
            zoom_factor = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
            image_tensor = tf.image.resize(
                image_tensor, 
                [int(self.target_size[1] * zoom_factor), int(self.target_size[0] * zoom_factor)]
            )
            # Crop back to original size
            image_tensor = tf.image.resize_with_crop_or_pad(
                image_tensor, self.target_size[1], self.target_size[0]
            )
        
        return image_tensor.numpy()


class DataLoader:
    """
    Handles loading and organizing plant image datasets
    """
    
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing plant species folders
            target_size: Target image size
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size)
        self.class_names = []
        
    def load_dataset(self, validation_split: float = 0.2, 
                    batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and validation datasets
        
        Args:
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        # Use TensorFlow's image_dataset_from_directory
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=self.target_size,
            batch_size=batch_size
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=self.target_size,
            batch_size=batch_size
        )
        
        # Store class names
        self.class_names = train_ds.class_names
        
        # Normalize pixel values
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Optimize for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def get_class_names(self) -> List[str]:
        """Get list of class names (plant species)"""
        if not self.class_names:
            # Scan directory for class folders
            self.class_names = sorted([
                d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d))
            ])
        return self.class_names
    
    def get_sample_images(self, num_samples: int = 9) -> List[Tuple[str, str]]:
        """
        Get sample images from each class
        
        Args:
            num_samples: Number of sample images to return
            
        Returns:
            List of (image_path, class_name) tuples
        """
        samples = []
        class_names = self.get_class_names()
        
        for class_name in class_names[:num_samples]:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    sample_path = os.path.join(class_dir, image_files[0])
                    samples.append((sample_path, class_name))
                    
        return samples