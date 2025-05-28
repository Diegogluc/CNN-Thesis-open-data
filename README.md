# CNN Classification Methods Comparison

A comprehensive comparative study of CNN architectures for 1D and 2D signal classification, implementing and evaluating methods from recent literature on the IASC benchmark dataset.

## 🎯 Overview

This project implements and compares several state-of-the-art CNN architectures for signal classification tasks. The study focuses on systematic evaluation using k-fold cross-validation to provide robust performance comparisons across different architectural approaches.

## 🏗️ Implemented Methods

- **Liu et al. (2020)**: Compact CNN architecture with tanh activation and minimal layers for efficient 1D signal processing
- **Rezende et al. (2020)**: Deep learning approach with dropout regularization and dense connectivity for robust classification
- **Azimi et al. (2020)**: Multi-block CNN with batch normalization and LeakyReLU activation for enhanced feature extraction
- **Park et al. (2020)**: 2D CNN architecture for spectral data classification using time-frequency representations

## 📊 Dataset

- **IASC Dataset**: Public benchmark dataset for structural health monitoring and vibration analysis
- **Data Types**: Both 1D time-series and 2D spectral representations
- **Classes**: 9 different structural conditions
- **Preprocessing**: Standardized preprocessing pipeline with configurable channel selection
- Preprocessed data available in `data/iasc_dataset/processed/`

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for faster training)
- 8GB+ RAM recommended

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-classification-comparison.git
cd cnn-classification-comparison

# Create virtual environment (recommended)
python -m venv cnn_env
source cnn_env/bin/activate  # On Windows: cnn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💡 Usage

### Quick Start
```bash
# Compare all 1D CNN models with default settings
python main.py

# Evaluate Park 2D model
python main.py --model park

# Custom configuration
python main.py --model all --epochs 50 --folds 10 --batch-size 32
```

### Command Line Options
```bash
python main.py --help

Options:
  --model {all,azimi,liu,rezende,park}  Which model(s) to evaluate (default: all)
  --epochs EPOCHS                      Number of training epochs (default: 30)
  --folds FOLDS                        Number of CV folds (default: 5)
  --batch-size BATCH_SIZE              Training batch size (default: 64)
  --gpu-memory GPU_MEMORY              GPU memory limit in MB (default: 7168)
```

### Programmatic Usage
```python
# Load data
from data.data_loader import load_iasc_data
X, y = load_iasc_data(data_type='1D')

# Load and create model
from models.baseline_models import create_liu_model
model = create_liu_model(input_shape=X.shape[1:])

# Evaluate with cross-validation
from models.model_utils import evaluate_model_kfold
results = evaluate_model_kfold(lambda: create_liu_model(X.shape[1:]), X, y)
```

## 📁 Project Structure

```
cnn-classification-comparison/
├── main.py                    # Main entry point
├── README.md                  # This file
├── requirements.txt           # Dependencies
│
├── models/                    # CNN implementations
│   ├── baseline_models.py     # Literature-based architectures
│   └── model_utils.py         # Training and evaluation utilities
│
├── data/                      # Data handling
│   ├── data_loader.py         # Data loading utilities
│   └── iasc_dataset/          # IASC dataset
│       └── processed/         # Preprocessed pickle files
│
├── scripts/                   # Individual component scripts
│   ├── compare_1d_models.py   # 1D model comparison
│   └── evaluate_park_model.py # 2D Park model evaluation
│
├── hyperparameter_tuning/     # Hyperparameter optimization
│   ├── iasc_tuning.py         # 1D models tuning
│   └── park_tuning.py         # Park model tuning
│
└── results/                   # Output files and logs
    ├── figures/               # Generated plots
    └── *.txt                  # Cross-validation results
```

## 🔬 Methodology

### Cross-Validation Strategy
- **K-Fold Cross-Validation**: 5-fold stratified cross-validation for robust performance estimation
- **Metrics**: Accuracy, Precision, and Loss tracking across all folds
- **Reproducibility**: Fixed random seeds for consistent results

### Model Training
- **Optimization**: Adam optimizer with architecture-specific learning rates
- **Loss Function**: Categorical cross-entropy for multi-class classification
- **Batch Processing**: Configurable batch sizes with default of 64
- **GPU Support**: Automatic GPU detection with memory limit configuration

### Evaluation Protocol
- **Performance Metrics**: Mean accuracy with standard deviation across folds
- **Statistical Analysis**: Comprehensive reporting of per-fold results
- **Model Comparison**: Side-by-side performance comparison across architectures

## 📈 Results

Results are automatically saved to the `results/` directory after each evaluation:

- **Individual Model Reports**: Detailed per-fold results in `kfold_[ModelName].txt`
- **Summary Statistics**: Mean accuracy ± standard deviation for each architecture
- **Performance Comparison**: Comparative analysis across all implemented methods

Example output format:
```
Azimi       - Accuracy:  87.45 (+- 2.31) - Loss: 0.3456
Liu         - Accuracy:  84.12 (+- 3.02) - Loss: 0.4123
Rezende     - Accuracy:  85.78 (+- 2.67) - Loss: 0.3889
Park        - Accuracy:  89.23 (+- 1.98) - Loss: 0.2967
```

## 🛠️ Development

### Adding New Models
1. Implement the model function in `models/baseline_models.py`
2. Add appropriate documentation following the existing pattern
3. Update the model selection in `scripts/compare_1d_models.py`

### Hyperparameter Tuning
Use the provided Keras Tuner scripts:
```bash
python hyperparameter_tuning/iasc_tuning.py    # For 1D models
python hyperparameter_tuning/park_tuning.py    # For Park model
```

## 📚 References

- LIU, T. et al. A data-driven damage identification framework based on transmissibility function datasets and one-dimensional convolutional neural networks: verification on a structural health monitoring benchmark structure. Sensors, v. 20, p. 1059, 2020.
- REZENDE, S. W. F. et al. Convolutional neural network and impedance-based SHM applied to damage detection. Engineering Research Express, v. 2, p. 035031, 2020.
- AZIMI, M.; PEKCAN, G. Structural health monitoring using extremely compressed data through deep learning. Computer-Aided Civil and Infrastructure Engineering,v. 35, p. 597–614, 2020.
- PARK, H. S. et al. Convolutional neural network-based safety evaluation method for structures with dynamic responses. Expert Systems with Applications, v. 158, p. 113634, 2020.

## 🤝 Contributing

This project is part of ongoing research. For questions or collaboration opportunities, please open an issue or contact the maintainer.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Technical Requirements

- **TensorFlow**: ≥2.10.1
- **Keras Tuner**: ≥1.4.7  
- **NumPy**: ≥1.26.4
- **Scikit-learn**: ≥1.5.1
- **Matplotlib**: ≥3.9.2
- **Pandas**: ≥2.2.2

## 📞 Support

If you encounter any issues:
1. Check the [Issues](../../issues) page for common problems
2. Ensure all dependencies are correctly installed
3. Verify GPU setup if using CUDA acceleration
4. Check data file paths and permissions

---

**Note**: This implementation focuses on reproducible research and clean code architecture. The models are implemented based on published literature for comparative analysis purposes.
