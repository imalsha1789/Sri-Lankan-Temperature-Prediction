# Sri Lankan Daily Maximum Temperature Prediction Using Ensemble Machine Learning Models

## Project Overview

This project focuses on predicting daily maximum temperatures in Sri Lanka using ensemble machine learning models.

The main objective is to develop and compare different ensemble learning approaches using historical Sri Lankan weather data and identify the models that provide better prediction performance.

A Streamlit dashboard is also developed to allow users to explore model performance and obtain daily maximum temperature predictions based on user-provided weather information.

**Important:** This project estimates temperature based on the input features provided by the user. It is not designed to provide future weather forecasts.

## Dataset

The project uses a historical Sri Lankan weather dataset containing the following features:

- Date
- City
- Latitude
- Longitude
- Maximum Temperature (°C)
- Minimum Temperature (°C)
- Precipitation (mm)
- Maximum Wind Speed (km/h)
- Mean Humidity (%)

**Dataset Source:**

https://github.com/dhananjaya-hbc/lanka-data-tracker

## Ensemble Machine Learning Models

Several ensemble learning approaches were implemented and evaluated.

### 1. Bagging

- Random Forest

### 2. Boosting

- XGBoost
- LightGBM
- CatBoost

### 3. Stacking

- Stacking Ensemble Model

The stacking model combines predictions from multiple base models using a meta-model to generate the final prediction.

### 4. Hybrid Model

- Hybrid Random Forest–Artificial Neural Network (RF–ANN)

This hybrid approach combines Random Forest and Artificial Neural Network techniques.

## Model Evaluation

The models were evaluated using the following performance metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted temperatures.
- **Root Mean Squared Error (RMSE):** Measures prediction errors while giving greater weight to larger errors.
- **R² Score:** Measures how well the model explains the variation in the target variable.

Hyperparameter tuning was performed using GridSearchCV with 3-fold cross-validation for selected models.

### Baseline Model Comparison Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Stacking | 0.806 | 1.045 | 0.901 |
| XGBoost | 0.818 | 1.063 | 0.898 |
| LightGBM | 0.819 | 1.062 | 0.898 |
| Random Forest | 0.839 | 1.093 | 0.892 |
| CatBoost | 0.858 | 1.109 | 0.889 |
| Hybrid RF + ANN | 0.872 | 1.131 | 0.884 |

The Stacking Ensemble achieved the best overall baseline test performance among the six evaluated models.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Dashboard Features

The Streamlit dashboard provides the following features:

- Project Overview
- Daily Maximum Temperature Prediction
- Model Performance Comparison
- MAE, RMSE and R² Score Visualization
- Feature Importance Analysis
- Project Information

## Project Structure

```text
ML_Assignment/
│
├── app.py
├── Sri_Lankan_Temperature_Prediction.ipynb
├── requirements.txt
├── README.md
│
├── dataset/
│   └── Weather dataset CSV files
│
├── stacking_model.pkl
│
├── model_comparison_mae.png
├── model_comparison_r2.png
├── model_comparison_rmse.png
├── random_forest_feature_importance.png
└── xgboost_feature_importance.png
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/imalsha1789/Sri-Lankan-Temperature-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Sri-Lankan-Temperature-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```bash
py -m streamlit run app.py
```

The dashboard will open in your web browser.

**Note:** The repository uses Git LFS for the trained stacking model. Git LFS must be installed to download the actual model file when cloning the repository.

## Generative AI Usage

Generative AI tools were used as supporting tools during project development for:

- Code assistance and debugging.
- Understanding machine learning concepts.
- Assistance with data analysis.
- Documentation and explanation of project components.

The project implementation, model experimentation, evaluation and final results were reviewed as part of the development process.

## Live Dashboard

https://sri-lankan-temperature-prediction.streamlit.app/

## GitHub Repository

https://github.com/imalsha1789/Sri-Lankan-Temperature-Prediction
