# Credit Card Fraud Detection Project

## Project Overview

This project implements a machine learning solution to detect fraudulent credit card transactions. Using a combination of transaction details, customer information, and merchant data, the model identifies potential fraud cases with high accuracy.

## Model Performance

- **Best F1 Score**: ~0.96 (5-fold cross-validation)
- **Model Type**: Random Forest with SMOTE
- **Data Balance**: Original data contains ~11.4% fraudulent transactions

## Features

The model uses six core features:

1. `amt`: Transaction amount
2. `trans_hour`: Hour of transaction
3. `trans_day`: Day of transaction
4. `age`: Customer age
5. `category_code`: Encoded transaction category
6. `gender`: Customer gender (encoded)

## Technical Implementation

### Data Preprocessing

1. **Time Processing**:

   - Converted transaction dates to datetime
   - Extracted hour and day information

2. **Age Calculation**:

   - Calculated precise age at transaction time
   - Considered month and day for accurate age computation

3. **Feature Encoding**:
   - Label encoded categorical variables
   - Binary encoded gender (M/F)

### Model Architecture

1. **Data Pipeline**:

   ```python
   Pipeline([
       ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
       ('classifier', RandomForestClassifier(
           n_estimators=220,
           random_state=42,
           class_weight=None,
           n_jobs=-1
       ))
   ])
   ```

2. **Key Components**:
   - SMOTE resampling with 0.3 sampling strategy
   - Random Forest with 220 trees
   - StandardScaler for feature normalization

### Validation Strategy

- 5-fold cross-validation
- Stratified sampling to maintain class distribution
- F1 score as primary metric

## Model Development Process

### Initial Approach

- Started with basic Random Forest
- Used standard feature engineering
- Identified class imbalance issue

### Improvements

1. **Data Balancing**:

   - Implemented SMOTE for minority class oversampling
   - Tested different sampling ratios
   - Selected 0.3 as optimal sampling strategy

2. **Model Tuning**:

   - Optimized number of trees (n_estimators=220)
   - Removed class weights due to SMOTE
   - Utilized all CPU cores for faster training

3. **Feature Selection**:
   - Started with comprehensive feature set
   - Identified most important features through feature importance analysis
   - Reduced to six core features for optimal performance

## Predictions Distribution

The model maintains a realistic prediction distribution:

- Legitimate transactions: ~87%
- Fraudulent transactions: ~13%

## Key Learnings

1. Feature importance analysis showed transaction amount and time as critical factors
2. SMOTE proved more effective than class weights for handling imbalance
3. Cross-validation ensures robust performance estimation

## Future Improvements

1. Consider adding feature interactions
2. Experiment with different sampling techniques
3. Test alternative models (XGBoost, LightGBM)
4. Implement feature selection based on domain knowledge

## Technical Requirements

- Python 3.x
- Required packages:
  ```
  pandas
  numpy
  scikit-learn
  imbalanced-learn
  ```

## Usage

1. Prepare data in CSV format
2. Run feature engineering
3. Execute model training
4. Generate predictions for new transactions

## File Structure

```
fraud_detection/
│
├── train.csv           # Training data
├── test.csv            # Test data
├── main.py            # Main script
└── submission.csv     # Predictions output
```
