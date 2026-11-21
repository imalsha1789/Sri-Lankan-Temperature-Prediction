# Sri Lankan Daily Maximum Temperature Prediction Using Ensemble Machine Learning Models

## Project Overview

This project focuses on predicting daily maximum temperatures in Sri Lanka using ensemble machine learning models.

Several machine learning approaches were developed and compared to identify their prediction performance using historical weather data.

## Dataset

The project uses a Sri Lankan weather dataset containing the following features:

- Date
- City
- Latitude
- Longitude
- Maximum Temperature (°C)
- Minimum Temperature (°C)
- Precipitation (mm)
- Maximum Wind Speed (km/h)
- Mean Humidity (%)

Dataset source:

https://github.com/dhananjaya-hbc/lanka-data-tracker

## Machine Learning Models

The following models were implemented:

1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost
5. Stacking Ensemble
6. Hybrid Random Forest–ANN

## Model Evaluation

The models were evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Hyperparameter tuning was performed using GridSearchCV with 3-fold cross-validation.

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

## Dashboard Features

- Project overview
- Daily maximum temperature prediction
- Model performance comparison
- Feature importance analysis
- Project information

## Project Structure

```text
ML_Assignment/
│
├── app.py
├── Sri_Lankan_Temperature_Prediction.ipynb
├── requirements.txt
├── dataset/
├── *.pkl
└── README.md
```

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
py -m streamlit run app.py
```

## Important Note

The dashboard estimates daily maximum temperature based on the input features provided by the user. It is not designed to provide future weather forecasts.

## Generative AI Usage

Generative AI tools were used as supporting tools during project development for code assistance, debugging, understanding machine learning concepts, data analysis, and documentation.