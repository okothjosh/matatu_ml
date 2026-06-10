# matatu_ml

Final year school project on a machine learning system that predicts demand and pricing of matatus in Kenya

## 🚌 Project Overview

This project develops a comprehensive machine learning system that:
- **Predicts demand** for matatu (mini-bus) services in Kenya
- **Optimizes pricing** strategies based on demand patterns and market conditions
- **Analyzes surge pricing** opportunities across different routes

## 📊 Features

- Data-driven insights from real matatu operations
- Advanced ML models (XGBoost, Neural Networks)
- Demand forecasting and trend analysis
- Revenue optimization through surge pricing simulation
- Interactive web dashboard (Streamlit)

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting models
- **TensorFlow/Keras** - Deep learning models
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework

## 📁 Project Structure

```
matatu_ml/
├── 01_data_ingestion.ipynb           # Data collection and loading
├── 02_preprocessing.ipynb             # Feature engineering and cleaning
├── 03_model_training_evaluation.ipynb # Traditional ML models
├── 04_XGBoost_model_training.ipynb   # Advanced gradient boosting
├── 05_surge_simulation.ipynb          # Pricing simulation and analysis
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
└── data/                              # Dataset and outputs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/okothjosh/matatu_ml.git
cd matatu_ml
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

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📖 Project Notebooks

### 1. Data Ingestion
- Loads raw matatu operational data
- Data validation and quality checks
- Exploratory overview of datasets

### 2. Preprocessing
- Missing value imputation
- Feature scaling and normalization
- Categorical encoding
- Feature engineering

### 3. Model Training & Evaluation
- Random Forest models
- Gradient Boosting
- Cross-validation
- Performance metrics (MAE, RMSE, R²)

### 4. XGBoost Training
- Hyperparameter tuning
- Advanced ensemble methods
- Model comparison and selection

### 5. Surge Simulation
- Surge pricing scenarios
- Revenue impact analysis
- Route-specific surge multipliers

## 📊 Dashboard Features

The Streamlit dashboard includes:

- **Home** - Project overview and key metrics
- **Data Insights** - Dataset statistics and visualizations
- **Model Performance** - Model comparison and evaluation metrics
- **Predictions** - Test set predictions with download option
- **Surge Analysis** - Route rankings and pricing scenarios
- **About** - Project details and documentation

## 📈 Key Results

- Demand prediction models with high accuracy
- Identified high-value surge pricing opportunities
- Route-specific pricing recommendations
- Revenue optimization strategies

## 👤 Author

**Joshua Okoth**

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or inquiries, please reach out through GitHub.

---

**Note**: This is a final year university project. Some data may be simulated or modified for demonstration purposes.
