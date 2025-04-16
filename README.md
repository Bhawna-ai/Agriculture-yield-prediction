# Crop Yield Prediction Model

A machine learning model that predicts crop yields using historical agricultural data. This project leverages Dask for efficient data processing and Random Forest for accurate predictions.

## 🎯 Features

- **Efficient Data Processing**: Uses Dask for parallel processing of large agricultural datasets
- **Advanced Feature Engineering**: Implements sophisticated features like:
  - Area-based features
  - Historical yield patterns
  - Seasonal patterns
  - Regional patterns
  - Interaction features
- **High Accuracy**: Achieves 91.16% R² score on test data
- **Visualization Tools**: Generates comprehensive visualizations including:
  - Feature importance plots
  - Actual vs Predicted yield plots
  - Yield distribution analysis
  - Correlation heatmaps

## 📊 Model Performance

The model demonstrates excellent performance metrics:
- R² Score: 0.9116 (91.16%)
- Cross-validation scores: [0.9072, 0.9064, 0.8852]
- RMSE: 274.02
- MAE: 17.68
- MAPE: 49.47%

## 🛠️ Technical Stack

- Python 3.x
- Dask (for parallel processing)
- Scikit-learn (for machine learning)
- Pandas & NumPy (for data manipulation)
- Matplotlib & Seaborn (for visualization)
- Joblib (for model persistence)

## 🖥️ Development Tools

This project was developed using:

- **Google Colab**: For cloud-based development and experimentation:
  - Free GPU/TPU access
  - Collaborative environment
  - Easy sharing of notebooks
  - Pre-installed ML libraries

- **GitHub**: For version control and collaboration:
  - Code repository management
  - Issue tracking
  - Pull request reviews
  - Project documentation

## 📋 Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  ```
  dask>=2023.3.0
  scikit-learn>=1.0.2
  pandas>=1.3.0
  numpy>=1.21.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  joblib>=1.0.0
  ```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop_yield_prediction.git
   cd crop_yield_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the project directory as `crop_yield_train.csv`

## 💻 Usage

1. Run the model:
   ```bash
   python best_model.py
   ```

2. The script will:
   - Set up Dask distributed processing
   - Load and preprocess the data
   - Train the model
   - Generate visualizations
   - Save the trained model and encoders

3. Output files:
   - `crop_yield_model.joblib`: Trained model
   - `encoders.joblib`: Label encoders
   - `feature_cols.joblib`: Feature columns
   - Visualization plots in PNG format

## 📈 Model Architecture

The model uses a Random Forest Regressor with the following optimized parameters:
- Number of trees: 150
- Maximum depth: 20
- Minimum samples per split: 5
- Minimum samples per leaf: 2
- Maximum features: 'sqrt'

## 🔍 Feature Engineering

The model incorporates several sophisticated features:
1. **Basic Area Features**
   - Area ratio relative to state average
2. **Historical Patterns**
   - Crop-specific historical yields
3. **Seasonal Patterns**
   - Crop-season specific yield patterns
4. **Regional Patterns**
   - State-level yield averages
5. **Interaction Features**
   - Area-state interactions
   - Crop-season interactions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any queries or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Special thanks to the agricultural data providers
- Appreciation to all contributors and maintainers of the project
