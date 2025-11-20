# SLOBnB First Draft
- Ian Ang | izang@calpoly.edu

## Model Performance
### Metrics Used & Why
**Root Mean Squared Error (RMSE)**
- Measures average prediction error in dollars but penalizes large errors more heavily due to the squaring

**Mean Absolute Error (MAE)**
- Also measures average prediction error in dollars but treats errors equally
- Think of it like: "on average, we're off by $X"

**R²**
- Explains the proportion of variance in the target variable that's predictable from the features

**Mean Absolute Percentage Error (MAPE)**
- Measures error as a percentage of actual price
Example:
    ```bash
    Actual: $100, Predicted: $120
    Error: $20
    Percentage error: $20/$100 = 0.20 = 20%
    
    Actual: $200, Predicted: $220  
    Error: $20
    Percentage error: $20/$200 = 0.10 = 10%
    
    MAPE = (20% + 10%) / 2 = 15%
    ```

### Baseline (predict mean all the time)
**Performance Metrics:**
| Metric | Train | Test |
|--------|-------|------|
| **RMSE** | $318.03 | $334.83 |
| **MAE** | $221.50 | $236.63 |
| **R²** | 0.0000 | -0.0012 |
| **MAPE** | 97.31% | 99.71% |

**Interpretation:**
- RMSE ~$334: Average error is 85% of the mean price
- MAE ~$237: On average, predictions are off by $237
- R² ≈ 0: Model explains no variance
- MAPE ~100%: Predictions are almost always 100% off the actual price


### Linear Regressor

**Performance Metrics:**

| Metric | Train | Test |
|--------|-------|------|
| **RMSE** | $207.49 | $198.06 |
| **MAE** | $139.13 | $137.70 |
| **R²** | 0.5743 | 0.6497 |
| **MAPE** | 47.80% | 44.94% |

**Improvement over Baseline:**
- RMSE: **41% better** ($136.77 improvement)
- MAE: **42% better** - reduced from $236.63 to $137.70 ($98.93 improvement)
- R²: **+0.65** (explains 65% of variance vs 0%)
- MAPE: **55% better** (from 100% to 45%)

**Learned Coefficients:**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Intercept | $46.42 | Base price |
| `bedrooms` | +$108.04 | Each bedroom adds $108 |
| `accommodates` | +$40.14 | Each guest capacity adds $40 |
| `rt_Hotel room` | +$307.70 | Hotel rooms add $308 |
| `rt_Private room` | -$71.67 | Private rooms reduce by $72 |
| `rt_Entire home/apt` | -$85.42 | Entire homes reduce by $85 |
| `rt_Shared room` | -$150.61 | Shared rooms reduce by $151 |

### Random Forest
**Hyperparameters:**
- `n_estimators`: 100 trees
- `max_depth`: Unlimited
- `min_samples_split`: 2
- `random_state`: 42

**Performance Metrics:**

| Metric | Train | Test |
|--------|-------|------|
| **RMSE** | $190.89 | $194.15 |
| **MAE** | $124.73 | $129.98 |
| **R²** | 0.6397 | 0.6634 |
| **MAPE** | 41.60% | 42.50% |

**Improvement over Baseline:**
- RMSE: **42% better** - reduced from $334.83 to $194.15 ($140.68 improvement)
- MAE: **45% better** - reduced from $236.63 to $129.98 ($106.65 improvement)
- R²: **+0.66** - explains 66% of variance (vs 0% for baseline)
- MAPE: **57% better** - reduced from 100% to 43%

**Improvement over Linear Regression:**
- RMSE: **2% better** - reduced from $198.06 to $194.15 ($3.91 improvement)
- MAE: **6% better** - reduced from $137.70 to $129.98 ($7.72 improvement)
- R²: **+0.01**
- MAPE: **5% better**

**Feature Importance Chart**
![image](https://hackmd.io/_uploads/HJS4eghe-l.png)

**Residual Plot**
![image](https://hackmd.io/_uploads/BJdXxehe-e.png)
- Points scattered mostly randomly around zero which is a good sign
- Most points within ±1 standard deviation

**Actual vs Predicted Plot**
![image](https://hackmd.io/_uploads/HyxBeg2e-l.png)
- Points cluster reasonably around the diagonal
- Some scatter at higher price ranges (>$500)
- A few points far from the line indicate some listings the model can't handle

## Strengths and weaknesses of the baseline
**Strengths:**
- Simple and fast to compute
- Provides a clear performance floor
- Good sanity check
    - If a complex model doesn't beat this, something is wrong

**Weaknesses:**
- Completely useless for actual predictions


## Possible reasons for errors or bias

Missing critical features
- I intentionally left out a lot of features for the sake of simplicity but I believe there's still a lot of signal that I should be able to pull from all of the other features available

Single snapshot of data
- I only used the 28 June 2025 snapshot of data but I still have 28 March 2025 and 31 December 2024 data
- I'm thinking more data could help smoothen out the noise


## Ideas for final report 

- I'm currently only using the 28 June 2025 snapshot of data so something on my radar is including the 28 March 2025 and 31 December 2024 data
    - Doing so will allow me to incorporate multi-season data but a challenge is ensuring proper ID-based splitting
- I started with a really basic set of features to really ensure I can start with a baseline and then systematically add and remove features while ensuring that doing so actually improves the performance. So adding new features is definitely up next
- Potentially experiment with more models but also tune the hyperparameters for my random forest model
- Figuring out if Santa Cruz dataset can be used on the AirBnB's in SLO
    - I haven't quite looked into it but a potential idea I was considering was scraping the AirBnB data on my own 
