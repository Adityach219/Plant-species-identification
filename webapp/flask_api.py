"""
Flask API for plant species identification
"""

from flask import Flask, request, jsonify, render_template_string
import os
import sys
import json
import numpy as np
from PIL import Image
import io
import base64

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.plant_classifier import PlantClassifier
from data.preprocessor import ImagePreprocessor

app = Flask(__name__)

# Global variables for model
classifier = None
preprocessor = None
class_names = []

# HTML template for simple web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Species Identification API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
        .result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
        .error { color: red; }
        .success { color: green; }
        .predictions { margin-top: 10px; }
        .prediction { padding: 5px; margin: 5px 0; background: white; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 Plant Species Identification API</h1>
        
        <div class="upload-form">
            <h3>Upload Plant Image</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <input type="number" name="top_k" value="5" min="1" max="10" placeholder="Number of predictions">
                <br><br>
                <button type="submit">Identify Plant</button>
            </form>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <h2>API Endpoints</h2>
        <ul>
            <li><strong>POST /predict</strong> - Upload image for prediction</li>
            <li><strong>GET /health</strong> - Check API health</li>
            <li><strong>GET /species</strong> - List supported species</li>
        </ul>
        
        <h3>Example Usage:</h3>
        <pre>
curl -X POST -F "image=@plant.jpg" -F "top_k=3" http://localhost:5000/predict
        </pre>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.querySelector('input[type="file"]');
            const topKInput = document.querySelector('input[name="top_k"]');
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            formData.append('image', fileInput.files[0]);
            formData.append('top_k', topKInput.value);
            
            document.getElementById('result').innerHTML = '<p>Processing...</p>';
            document.getElementById('result').style.display = 'block';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let resultHtml = '';
                
                if (data.error) {
                    resultHtml = '<p class="error">Error: ' + data.error + '</p>';
                } else {
                    resultHtml = '<h3 class="success">Results:</h3>';
                    resultHtml += '<div class="predictions">';
                    
                    data.predictions.forEach((pred, idx) => {
                        resultHtml += '<div class="prediction">';
                        resultHtml += '<strong>' + (idx + 1) + '. ' + pred.species + '</strong> ';
                        resultHtml += '(' + (pred.confidence * 100).toFixed(1) + '%)';
                        resultHtml += '</div>';
                    });
                    
                    resultHtml += '</div>';
                    resultHtml += '<p><strong>Processing time:</strong> ' + data.processing_time.toFixed(2) + ' seconds</p>';
                }
                
                document.getElementById('result').innerHTML = resultHtml;
            })
            .catch(error => {
                document.getElementById('result').innerHTML = '<p class="error">Error: ' + error + '</p>';
            });
        };
    </script>
</body>
</html>
"""


def load_model_and_classes(model_path, classes_path=None):
    """Load model and class names"""
    global classifier, preprocessor, class_names
    
    classifier = PlantClassifier()
    classifier.load_model(model_path)
    preprocessor = ImagePreprocessor()
    
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        classifier.class_names = class_names
    else:
        class_names = []


@app.route('/')
def index():
    """Serve simple web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if classifier is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'Plant identification API is running',
        'model_loaded': True,
        'num_classes': len(class_names) if class_names else 'unknown'
    })


@app.route('/species', methods=['GET'])
def list_species():
    """List supported species"""
    if not class_names:
        return jsonify({
            'species': [],
            'message': 'Class names not available'
        })
    
    return jsonify({
        'species': class_names,
        'count': len(class_names)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict plant species from uploaded image"""
    import time
    start_time = time.time()
    
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    # Get top_k parameter
    top_k = int(request.form.get('top_k', 5))
    top_k = max(1, min(top_k, 20))  # Limit between 1 and 20
    
    try:
        # Read and preprocess image
        image = Image.open(file.stream)
        processed_image = preprocessor.preprocess_pil_image(image)
        
        # Make prediction
        predictions = classifier.predict_top_k(processed_image, k=top_k)
        
        # Format results
        results = []
        for species, confidence in predictions:
            results.append({
                'species': species,
                'confidence': float(confidence)
            })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'predictions': results,
            'processing_time': processing_time,
            'image_size': image.size,
            'top_k': top_k
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Species Identification Flask API')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--classes', type=str,
                       help='Path to class names JSON file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        sys.exit(1)
    
    if args.classes and not os.path.exists(args.classes):
        print(f"Warning: Classes file '{args.classes}' does not exist")
    
    # Load model
    print("Loading model...")
    try:
        load_model_and_classes(args.model, args.classes)
        print("Model loaded successfully!")
        if class_names:
            print(f"Loaded {len(class_names)} species classes")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Start server
    print(f"Starting Flask API server...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API endpoints:")
    print(f"  - POST /predict - Upload image for prediction")
    print(f"  - GET /health - Health check")
    print(f"  - GET /species - List species")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()