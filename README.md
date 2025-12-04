# SLObnb: Predicting Airbnb Prices in College Beach Towns

A machine learning project that predicts nightly prices for **Airbnb listings in Santa Cruz County** using **systematic feature engineering** and **gradient boosting** (XGBoost).


## Background

This project was developed for **CSC466 - Knowledge Discovery from Data** (Fall 2025) at Cal Poly SLO, taught by Professor Lucas Pierce.


## Dataset

* **Source:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) (June 2025 snapshot)
* **Location:** Santa Cruz County, California
* **Size:** 1,749 listings
* **Price Range:** $27 - $2,999 per night
* **Split:** 80% train (1,236 listings) / 20% test (309 listings)


## Results

| Model | Test R² | MAE | MdAE | RMSE | Overfitting Gap |
|-------|---------|------|-----|------|-----------------|
| Baseline (Dummy) | -0.001 | $236.63 | $179.31 | $334.83 | 0.0012 |
| Linear Regression | 0.7055 | $128.98 | $95.82 | $181.61 | 0.0780 |
| Random Forest | 0.7283 | $116.15 | $77.96 | $175.04 | 0.0714 |
| **XGBoost** | **0.7324** | **$115.48** | **$76.90** | **$173.12** | **0.0660** |


## Tech Stack

* **Python 3.x**
* **scikit-learn** (Linear Regression, Random Forest, DummyRegressor)
* **XGBoost** (gradient boosting)
* **pandas** (data manipulation)
* **NumPy** (numerical computing)
* **matplotlib** (visualization)
* **Jupyter Notebook** (interactive analysis)

## Project Structure

```
├── data/
│   ├── 28june2025listings.csv       
│   ├── 28june2025calendar.csv       
│   ├── 28june2025reviews.csv        
│   └── ...
├── models/
│   ├── preprocessing.py             # Feature engineering pipeline
│   ├── evaluation.py                # Metric calculation functions
│   ├── baseline_model.py            # Dummy mean predictor
│   ├── linear_model.py              # Linear Regression
│   ├── random_forest_model.py       # Random Forest
│   ├── xgboost_model.py             # XGBoost
│   ├── transfer_learning.py         # Transfer learning experiments
│   └── visualization.py             # Plotting utilities
├── eda/
│   ├── eda_summary.md               
│   └── eda.ipynb                    # Exploratory data analysis
├── figures/
├── CSC466 Final Project Presentation             
├── first_draft.md                  
├── final_report.md                  # Complete technical writeup
└── README.md                        
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ian-ang-zhihan/slobnb.git
cd slobnb

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows: venv\Scripts\activate
                       # On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib
```

## How to Run

```bash
# Run XGBoost model (best performer)
python models/xgboost_model.py

# Run other models
python models/linear_model.py
python models/random_forest_model.py

# Run baseline comparison
python models/baseline_model.py
```

**Expected Output:**
- Model performance metrics (R², RMSE, MAE, MdAE, MAPE)
- Feature importance rankings
- Train vs test comparison

## Acknowledgments

* **Professor Lucas Pierce** — Course instructor (CSC466, Cal Poly SLO)
* **Inside Airbnb** — Open data source for Airbnb listings
* **GitHub Copilot, ChatGPT, and Claude** — AI assistants for debugging and documentation

