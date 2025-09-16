"""
Streamlit web application for plant species identification
"""

import streamlit as st
import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.plant_classifier import PlantClassifier
from data.preprocessor import ImagePreprocessor


@st.cache_resource
def load_model_and_classes(model_path, classes_path=None):
    """Load model and class names (cached for performance)"""
    classifier = PlantClassifier()
    classifier.load_model(model_path)
    
    class_names = []
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        classifier.class_names = class_names
    
    return classifier, class_names


def create_confidence_chart(predictions):
    """Create a horizontal bar chart for top predictions"""
    if not predictions:
        return None
    
    species = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(species, confidences, color='green', alpha=0.7)
    
    # Add percentage labels on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{conf:.1%}', va='center', fontweight='bold')
    
    ax.set_xlabel('Confidence')
    ax.set_title('Top Plant Species Predictions')
    ax.set_xlim(0, 1)
    
    # Reverse order to show highest confidence at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Plant Species Identification",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Plant Species Identification")
    st.markdown("Upload an image of a plant to identify its species using deep learning!")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model file selection
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="models/plant_classifier_custom_best.h5",
        help="Path to the trained model file"
    )
    
    classes_path = st.sidebar.text_input(
        "Classes Path", 
        value="models/plant_classifier_custom_classes.json",
        help="Path to the class names JSON file"
    )
    
    # Check if model files exist
    model_exists = os.path.exists(model_path)
    classes_exists = os.path.exists(classes_path) if classes_path else True
    
    if not model_exists:
        st.error(f"Model file not found: {model_path}")
        st.info("Please train a model first using the training script or provide a valid model path.")
        return
    
    if classes_path and not classes_exists:
        st.warning(f"Classes file not found: {classes_path}")
        st.info("Predictions will show class indices instead of species names.")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            classifier, class_names = load_model_and_classes(model_path, classes_path)
        
        st.sidebar.success("✅ Model loaded successfully!")
        if class_names:
            st.sidebar.info(f"📊 {len(class_names)} species classes available")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Plant Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear image of a plant for species identification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
            
    with col2:
        st.header("Identification Results")
        
        if uploaded_file is not None:
            try:
                # Preprocess image
                preprocessor = ImagePreprocessor()
                processed_image = preprocessor.preprocess_pil_image(image)
                
                # Make prediction
                with st.spinner("Identifying plant species..."):
                    top_predictions = classifier.predict_top_k(processed_image, k=5)
                
                # Display results
                if top_predictions:
                    # Top prediction
                    top_species, top_confidence = top_predictions[0]
                    
                    st.success(f"🌿 **Identified Species:** {top_species}")
                    st.metric("Confidence", f"{top_confidence:.1%}")
                    
                    # Confidence level indicator
                    if top_confidence >= 0.8:
                        st.success("🎯 High confidence prediction")
                    elif top_confidence >= 0.6:
                        st.warning("⚠️ Medium confidence prediction")
                    else:
                        st.error("❓ Low confidence prediction")
                    
                    # Detailed results
                    st.subheader("Top 5 Predictions")
                    
                    # Create confidence chart
                    fig = create_confidence_chart(top_predictions)
                    if fig:
                        st.pyplot(fig)
                    
                    # Detailed table
                    st.subheader("Detailed Results")
                    results_data = []
                    for i, (species, confidence) in enumerate(top_predictions, 1):
                        results_data.append({
                            "Rank": i,
                            "Species": species,
                            "Confidence": f"{confidence:.2%}"
                        })
                    
                    st.table(results_data)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 How to Use")
        st.markdown("""
        1. Upload a clear image of a plant
        2. Wait for the AI model to process
        3. View the predicted species and confidence
        4. Check the top 5 predictions for alternatives
        """)
    
    with col2:
        st.subheader("📸 Tips for Best Results")
        st.markdown("""
        - Use high-quality, well-lit images
        - Focus on distinctive plant features
        - Include leaves, flowers, or fruits
        - Avoid blurry or dark images
        """)
    
    with col3:
        st.subheader("🔧 Model Information")
        if class_names:
            st.markdown(f"""
            - **Species Classes:** {len(class_names)}
            - **Model Type:** Deep Learning CNN
            - **Input Size:** 224x224 pixels
            - **Framework:** TensorFlow/Keras
            """)
        else:
            st.markdown("""
            - **Model Type:** Deep Learning CNN
            - **Input Size:** 224x224 pixels
            - **Framework:** TensorFlow/Keras
            """)
    
    # Sample images section
    if st.checkbox("Show Sample Plant Images"):
        st.subheader("🖼️ Sample Plant Images")
        st.info("Here are examples of the types of plant images that work well with this system:")
        
        # Create placeholder for sample images
        sample_cols = st.columns(3)
        sample_descriptions = [
            "Clear leaf structure with good lighting",
            "Flower with distinctive features visible",
            "Full plant view showing overall structure"
        ]
        
        for i, (col, desc) in enumerate(zip(sample_cols, sample_descriptions)):
            with col:
                st.markdown(f"**Example {i+1}:**")
                st.info(desc)
                # You could add actual sample images here if available
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🌱 **Plant Species Identification System** | "
        "Built with Streamlit and TensorFlow | "
        "For educational and research purposes"
    )


if __name__ == "__main__":
    main()