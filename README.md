# 🏠 California Housing Price Prediction

A machine learning project that predicts median house prices in California districts using demographic, geographic, and housing features from the classic **California Housing Prices** dataset (Kaggle).

## 📋 Overview

This project builds and compares multiple regression models to predict `median_house_value` based on features like location (latitude/longitude), number of rooms, population, median income, and proximity to the ocean. The workflow covers data cleaning, feature engineering, model training, and evaluation using standard regression metrics.

## 📊 Dataset

- **Source:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) on Kaggle
- **Size:** ~20,640 rows, 10 columns
- **Target variable:** `median_house_value`

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms`, `total_bedrooms` | Total rooms/bedrooms in the block |
| `population`, `households` | Population and household counts |
| `median_income` | Median income of households (in tens of thousands of USD) |
| `ocean_proximity` | Categorical distance to ocean (`INLAND`, `NEAR BAY`, `<1H OCEAN`, etc.) |
| `median_house_value` | **Target** — median house value in USD |

## 🧹 Data Cleaning & Preprocessing

- Missing values in `total_bedrooms` imputed using the **median**
- Duplicate rows removed
- `ocean_proximity` one-hot encoded
- Numeric features scaled with `StandardScaler` (for the linear model)

## 🛠️ Feature Engineering

Three engineered ratio features were added, which correlate more strongly with price than their raw counterparts:

```python
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
```

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline, trained on scaled features |
| Decision Tree Regressor | Simple non-linear baseline |
| Random Forest Regressor | Ensemble, handles feature interactions well |
| Gradient Boosting Regressor | Typically strongest performer on tabular data |

## 📈 Evaluation Metrics

Models are evaluated using:
- **RMSE** (Root Mean Squared Error) — average error in USD, penalizes large errors more
- **MAE** (Mean Absolute Error) — average error in USD, robust to outliers
- **R²** (R-squared) — proportion of variance explained by the model

5-fold cross-validation is used to confirm results are stable across different data splits.

## 🔍 Key Findings

- `median_income` is by far the strongest predictor of house price
- Location-derived features (latitude/longitude, `ocean_proximity`) are highly influential — coastal areas command higher prices
- The dataset has a known artifact: house values are capped near **$500,000**, which affects model accuracy at the high end
- Tree-based ensemble models (Random Forest / Gradient Boosting) outperform Linear Regression due to non-linear relationships in the data

## 📁 Repository Structure

```
california-housing-price-prediction/
├── housing.csv                     # Dataset (from Kaggle)
├── housing_price_prediction.ipynb  # Main notebook: cleaning, EDA, training, evaluation
├── housing_price_model.joblib      # Saved final model
├── housing_scaler.joblib           # Saved StandardScaler (for Linear Regression)
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/california-housing-price-prediction.git
   cd california-housing-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `housing.csv` from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and place it in the project root.

4. Run the notebook:
   ```bash
   jupyter notebook housing_price_prediction.ipynb
   ```

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## 🚀 Future Improvements

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Try XGBoost / LightGBM for potentially stronger performance
- Deploy as an interactive Streamlit app (input features via sliders → live price prediction)
- Address the $500k price-cap artifact (e.g., exclude or separately model capped rows)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋 Author

**Ahmad Ali**
📧 meahmadali5at@gmail.com
