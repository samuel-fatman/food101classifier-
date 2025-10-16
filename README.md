# Food101 Classifier 🍕🍔🍎

A deep learning-powered food image classifier built with PyTorch and ResNet50, capable of identifying 101 different food categories. The project includes both training scripts and a user-friendly web interface powered by Gradio.

## 🚀 Features

### Core Functionality
- **Food Classification**: Classifies images into 101 different food categories from the Food-101 dataset
- **Pre-trained Model**: Uses ResNet50 architecture with transfer learning for accurate predictions
- **Web Interface**: Interactive Gradio web app for easy image upload and classification
- **Docker Support**: Fully containerized application with Docker and Docker Compose
- **GPU Support**: Automatic CUDA detection for accelerated inference
- **Top-K Predictions**: Returns top 5 most likely food categories with confidence scores

### Model Features
- **Transfer Learning**: Fine-tuned ResNet50 model pre-trained on ImageNet
- **Data Augmentation**: Comprehensive augmentation including rotation, flipping, and color jittering
- **Flexible Training**: Supports both full model training and fine-tuning of specific layers
- **Multiple Model Formats**: Supports loading full model checkpoints and state dictionaries

### Technical Features
- **Image Preprocessing**: Automatic image resizing, normalization, and format conversion
- **Label Mapping**: JSON-based label mapping for human-readable food category names
- **Error Handling**: Robust error handling for model loading and image processing
- **Cross-platform**: Compatible with Windows, macOS, and Linux

## 📁 Project Structure

```
food101classifier/
├── app.py                          # Main Gradio web application
├── app_previous.py                 # Previous version with advanced model loading
├── model.py                        # Training script with full pipeline
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml              # Docker Compose orchestration
├── requirements.txt                # Python dependencies
├── food101_labels.json             # Food category labels mapping
├── food101_resnet50_weights.pth    # Pre-trained model weights
├── food101_resnet50_full.pth       # Full model checkpoint
├── food101_resnet50_state_dict.pth # Model state dictionary
└── README.md                       # This file
```

## 🛠️ Installation & Setup

### Method 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone (https://github.com/samuel-fatman/food101classifier-.git)
   cd food101classifier
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:7860`

### Method 2: Local Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open `http://localhost:7860` in your browser

## 🎯 Usage

### Web Interface
1. Open the Gradio interface in your browser
2. Upload a food image using the file upload area
3. Click "Submit" to get predictions
4. View the top 5 food categories with confidence scores

### Supported Food Categories
The model can classify 101 different food types including:
- **Appetizers**: apple_pie, beet_salad, beef_carpaccio, beef_tartare
- **Main Dishes**: bibimbap, baby_back_ribs, chicken_curry, fish_and_chips
- **Desserts**: baklava, beignets, bread_pudding, cannoli
- **And many more!**

See `food101_labels.json` for the complete list of supported categories.

## 🧠 Model Architecture

### ResNet50 Transfer Learning
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Architecture**: Deep residual network with 50 layers
- **Fine-tuning**: Last layer + Layer4 block trainable
- **Output**: 101-class classification

### Training Configuration
- **Dataset**: Food-101 from Hugging Face datasets
- **Batch Size**: 32
- **Learning Rate**: 1e-4 with StepLR scheduler
- **Epochs**: 10 (configurable)
- **Validation Split**: 10% of training data
- **Optimizer**: Adam optimizer

### Data Augmentation
- **Training**:
  - Random resized crop (224x224)
  - Random horizontal flip
  - Random rotation (30°)
  - Color jittering (brightness, contrast, saturation, hue)
  - ImageNet normalization
- **Validation/Test**:
  - Resize to 224x224
  - ImageNet normalization

## 📊 Model Performance

The model achieves competitive accuracy on the Food-101 dataset through:
- Transfer learning from ImageNet
- Comprehensive data augmentation
- Fine-tuning strategy focusing on relevant layers
- Proper train/validation/test splits

## 🔧 Training Your Own Model

To train a new model from scratch:

```bash
python model.py
```

This will:
1. Download the Food-101 dataset from Hugging Face
2. Set up data loaders with augmentation
3. Initialize ResNet50 with transfer learning
4. Train for the specified number of epochs
5. Save the trained model weights
6. Evaluate on the test set

## 🐳 Docker Configuration

### Dockerfile Features
- **Base Image**: Python 3.10 slim
- **Port**: Exposes port 7860 for Gradio
- **Optimization**: Layered caching for faster builds
- **Dependencies**: Minimal required packages

### Docker Compose Features
- **Port Mapping**: Maps container port 7860 to host
- **Volume Mount**: Live code updates during development
- **Auto-restart**: Automatically restarts on failure

## 🔍 Technical Details

### Image Processing Pipeline
1. **Input**: PIL Image or numpy array
2. **Conversion**: Automatic RGB conversion
3. **Preprocessing**: Resize, crop, normalize
4. **Tensor**: Convert to PyTorch tensor
5. **Inference**: Forward pass through ResNet50
6. **Output**: Softmax probabilities for 101 classes

### Model Loading Strategy
- **Primary**: Load state dictionary weights
- **Fallback**: Support for full model checkpoints
- **Compatibility**: Handle both `module.` prefixed and regular state dicts
- **Device**: Automatic CPU/GPU selection

### Error Handling
- Graceful handling of missing label files
- Robust model loading with multiple fallback strategies
- Proper device management for CPU/GPU compatibility
- Image format validation and conversion

## 📋 Requirements

- **Python**: 3.8+ (3.10 recommended)
- **PyTorch**: Latest stable version
- **Torchvision**: For model architecture and transforms
- **Gradio**: For web interface
- **Pillow**: For image processing
- **Datasets**: For Food-101 dataset loading (training only)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source. Please check the repository for license details.

## 🙋 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check existing documentation
- Review the code comments for detailed explanations

## 🎉 Acknowledgments

- **Food-101 Dataset**: ETH Zurich for the comprehensive food dataset
- **PyTorch Team**: For the excellent deep learning framework
- **Gradio Team**: For the user-friendly web interface framework
- **ResNet**: Microsoft Research for the ResNet architecture

---

**Happy Food Classification! 🍽️**
