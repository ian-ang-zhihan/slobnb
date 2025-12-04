# SLObnb: Predicting Airbnb Prices in College Beach Towns
**Ian Ang** | CSC466 - Knowledge Discovery from Data | Prof. Lucas Pierce | December 3, 2025

## Introduction

Both Airbnb users and new Airbnb hosts face a similar pricing problem. As a user, how do I know if a listing is fairly or reasonably priced? As a new Airbnb host, what price should I charge? Setting prices too high risks no bookings; too low and leave significant revenue on the table. Without comparable market data or pricing expertise, both users and hosts rely on guesswork or competitor browsing, leading to suboptimal decisions that can cost hundreds of dollars.

This project predicts Airbnb listing prices in Santa Cruz County using machine learning. Through systematic feature engineering, XGBoost achieved R² = 0.7324 (73% variance explained) with RMSE = $173 on 1,739 listings ranging from $27 to $2,999 per night. Results demonstrate that systematic feature engineering and hyperparameter tuning, provide users and hosts with data-driven decision making and pricing guidance. A future plan is to transfer these learnings over and apply them to San Luis Obispo County should that data become available.

## Methods

The dataset consists of 1,739 Airbnb listings from Santa Cruz County, with nightly prices ranging from $27 to $2,999. The data was split into training (1,236 listings, 80%) and testing (309 listings, 20%) sets.

Feature engineering proceeded through iterative phases, each building upon insights from the previous stage. The baseline model used only three intuitive features: room type (entire home, private room, shared room), max number of guests accommodated, and number of bedrooms. This simple model achieved R² = 0.65, establishing a strong foundation but leaving substantial unexplained variance. A quick addition of number of beds was done right after this.

Parsing bathroom features increased R² by 6.0 percentage points to 0.71. Airbnb's bathroom information exists in both numeric and text formats, with the text containing critical distinctions like "1.5 shared baths" versus "2 private baths." Using regular expressions, I extracted two new features: `bathrooms_count` (numeric quantity) and `bathroom_type` (private, shared, or standard).

Phase 3 added quality signals including review scores rating, and superhost status, maintaining R² at 0.71. The final phase tackled the amenity dimensionality problem through aggregation. To avoid 1,454 spare amenity binaries, I created 11 category scores (parking, kitchen, entertainment, climate control, laundry, safety, convenience, toiletries, beach access, pet friendliness) by counting keyword matches within each category. This category-based scoring preserved amenity information while compressing 1,454 features into 11, increasing R² to 0.7324.

Both the initial Random Forest and XGBoost models with default parameters possessed overfitting issues. To prevent this, I manually tuned the hyperparameters and reached an acceptable train-test gap of 0.0714 and 0.0660 respectively.

## Results

XGBoost achieved the best performance on the held-out test set while maintaining excellent generalization. Random Forest performed comparably while Linear Regression lagged slightly behind. All three models significantly outperformed the baseline mean predictor, which achieved negative R² due to worse-than-average predictions.

| Model | Test R² | MAE | MdAE | RMSE | Overfitting Gap |
|-------|---------|------|-----|------|-----------------|
| Baseline (Dummy) | -0.001 | $236.63 | $179.31 | $334.83 | 0.0012 |
| Linear Regression | 0.7055 | $128.98 | $95.82 | $181.61 | 0.0780 |
| Random Forest | 0.7283 | $116.15 | $77.96 | $175.04 | 0.0714 |
| XGBoost | 0.7324 | $115.48 | $76.90 | $173.12 | 0.0660 |

Feature importance analysis revealed that capacity features overwhelmingly drive pricing decisions. The top three features: `accommodates` (37%), `bedrooms` (26%), and `bathrooms_count` (14%) collectively account for 77% of the model's decision-making. This finding aligns with my intuition that hosts price primarily based on how many guests a property can comfortably house, with amenities and quality signals playing secondary roles.

The model seems to have hit a practical ceiling. The missing 27% of explainability likely comes from factors not captured in the features such as seasonal pricing, precise location and neighborhood desirability, photo quality and aesthetics, and unavoidable human pricing noise from hosts’ personal judgments.

## Conclusion

The model offers three practical applications for Airbnb's ecosystem. First, it provides new hosts with data-driven starting prices based solely on property features, which they can subsequently adjust based on actual booking demand and market feedback. Second, feature importance analysis reveals which amenities offer the best return on investment—for example, adding parking or upgrading kitchen appliances may justify higher prices more than entertainment options. Third, the model serves as a sanity check for both users and hosts to flag potential pricing errors.

While higher R² remains unattainable with current features, R² = 0.7324 is sufficient for decision support. The model augments human judgment by helping new hosts avoid egregious errors while learning the market.
