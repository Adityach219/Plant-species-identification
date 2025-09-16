# Plant Species Identification

A comprehensive machine learning system for identifying plant species from images using deep learning techniques.

## Features

- **Deep Learning Model**: CNN-based architecture for accurate plant species classification
- **Data Preprocessing**: Automated image preprocessing and augmentation
- **Web Interface**: User-friendly web application for plant identification
- **CLI Interface**: Command-line tool for batch processing
- **Model Training**: Complete pipeline for training custom models
- **Docker Support**: Easy deployment with containerization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Adityach219/Plant-species-identification.git
cd Plant-species-identification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Web Interface

1. Start the web application:
```bash
streamlit run webapp/app.py
```

2. Open your browser and navigate to `http://localhost:8501`
3. Upload a plant image and get instant species identification

### Using the CLI

```bash
python src/plant_identifier/predict.py --image path/to/plant_image.jpg
```

### Training a Custom Model

```bash
python src/plant_identifier/train.py --data_dir data/training --epochs 50
```

## Project Structure

```
Plant-species-identification/
├── src/
│   ├── plant_identifier/        # Core ML modules
│   ├── data/                    # Data processing utilities
│   ├── models/                  # Model architectures
│   └── utils/                   # Helper functions
├── webapp/                      # Web application
├── data/                        # Dataset storage
├── config/                      # Configuration files
├── tests/                       # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Architecture

The system uses a Convolutional Neural Network (CNN) based on proven architectures like ResNet and EfficientNet, optimized for plant species classification. The model is trained on diverse plant datasets to ensure robust performance across different species.

## Supported Plant Species

The model currently supports identification of common plant species including:
- Trees (Oak, Maple, Pine, etc.)
- Flowers (Rose, Tulip, Sunflower, etc.)
- Shrubs and bushes
- Garden plants
- Wild plants

*Note: The model can be extended to support additional species by retraining with new datasets.*

## API Documentation

### REST API Endpoints

- `POST /predict` - Upload image and get species prediction
- `GET /health` - Check API health status
- `GET /species` - List supported species

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Plant datasets from various botanical databases
- TensorFlow and Keras for deep learning framework
- Streamlit for web interface development