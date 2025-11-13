# EDA Summary
- Ian Ang | izang@calpoly.edu

## What is your dataset and why did you choose it?
**Dataset:** Inside Airbnb - Santa Cruz County
**Source:** http://insideairbnb.com/get-the-data/

**Why this dataset:** I was unable to find any AirBnB data for San Luis Obispo data, so I chose Santa Cruz County since it was available in the Inside AirBnB website and I assume Santa Cruz County is the closest coastal California location with similar demographics and tourism patterns (college town, beach destination).

**Structure:**
- `listings.csv`: 1,739 listings with 79 columns
- `reviews.csv`: 160,945 reviews with 6 columns
- `calendar.csv`: 634,735 availability records with 7 columns
- `neighbourhoods.csv`: 5 neighborhoods

**Temporal Data:** Used June 28, 2025 snapshot as primary dataset. Also explored Dec 2024 and Mar 2025 snapshots for seasonality analysis.

## What did you learn from your EDA?
### Key Variables
- **Target:** `price` (continuous, cleaned from string format "$X.XX" → float)
- **Potential strong predictors:** `room_type`, `accommodates`, `bedrooms`, `neighbourhood_cleansed`
- **Potential interaction signals:** Review metrics (`reviews_per_month` has weak negative correlation with price: -0.086)

### Data Volume
28 June 2025
- **Listings:** 1,739 rows
- **Reviews:** 160,945 rows
- **Calendar:** 634,735 rows

28 March 2025
- **Listings:** 1,684 rows
- **Reviews:** 154,150 rows
- **Calendar:** 613,431 rows

31 December 2024
- **Listings:** 1,703 rows
- **Reviews:** 153,725 rows
- **Calendar:** 620,390 rows

### Missingness
- Core features like `price`, `room_type`, `accommodates` look to be complete
- Review-related fields: ~12% missing
- Other fields also seem to contain different percentages of missing values (e.g. `host_response_rate` = ~13%, etc) but may not be too important of a feature just yet

### Potential Target / Interaction Signals
- **`room_type`:** Clear price stratification (Entire home/apt > Private room > Shared room)
- **`accommodates`:** Positive correlation with price (more capacity = higher price)
- **`reviews_per_month`:** Weak negative correlation (-0.086) suggests cheaper listings get more bookings
- **Seasonality:** Price increases ~23% from spring (Mar) to summer (Jun) in stable listings

### Visualizations

![image](https://hackmd.io/_uploads/SJO1xZXgWe.png)

![image](https://hackmd.io/_uploads/Hkm1gW7x-g.png)

![image](https://hackmd.io/_uploads/HkTAy-QeZx.png)

![image](https://hackmd.io/_uploads/HyHRJ-QlZl.png)

![image](https://hackmd.io/_uploads/SkEaJWmeWg.png)

![image](https://hackmd.io/_uploads/Sy03yZ7xZx.png)

![image](https://hackmd.io/_uploads/rkBh1bmeWx.png)


### Feature Ideas
- **Temporal:** Season indicator (summer premium), day-of-week effects
- **Text extraction:** Amenity counts (WiFi, parking, kitchen), keyword analysis from descriptions
- **Derived metrics:** Price per person, host experience
- **Geographic:** Distance to beach/downtown, neighborhood price averages

### Potential Challenges
- **Class imbalance:** ~83% of listings are "Entire home/apt
- **Outlier handling:** Removed extreme outliers (>$50,000/night data errors), but luxury listings ($1000+) remain valid but rare
- **Transfer learning:** Will Santa Cruz patterns generalize to San Luis Obispo effectively?
- **Text features:** Many user-generated text fields need NLP processing

## What issues or open questions remain?
- Where can I find San Luis Obispo County AirBnB data or use the Santa Cruz County data in some way and ensure that it's effective and actually applies to SLO county?
    - Can I transfer learnings from Santa Cruz to SLO effectively? 
- What is seasonality like? How can I use this in my model?
- Will calendar availability be an important feature?
- Which of the 79 features actually contribute to predictions? Need feature importance analysis.
- There are many text fields in the data that are user inputed (e.g. reviews, neighborhood overview, host description, etc). I wonder if that might be a path worth pursuing in terms of turning it into effective features
- There are hotel rooms?!? In AirBnb??
- How am I going to evaluate if my predictions are good? What metric do I use?
