# Credit Card Fraud Detection Project

## Project Overview

This project implements a machine learning solution for credit card fraud detection using a Random Forest model with SMOTE for handling class imbalance. The model achieved an impressive F1 score of 0.974, demonstrating high accuracy in identifying fraudulent transactions.

## Model Performance

- **Average F1 Score**: 0.9740 (±0.0017)
- **Cross-validation F1 Scores**:
  - Fold 1: 0.97342
  - Fold 2: 0.97328
  - Fold 3: 0.97322
  - Fold 4: 0.97438
  - Fold 5: 0.97546
- **Prediction Distribution**:
  - Normal Transactions: 88.70%
  - Fraudulent Transactions: 11.30%

## Feature Importance

1. Transaction Amount (0.529642)

   - Most significant feature
   - Contributes over 52% to the model's decisions

2. Time-related Features

   - Transaction Hour (0.180485)
   - Transaction Day (0.156113)
   - Combined contribution of ~33.7%

3. Category Code (0.084461)

   - Fourth most important feature
   - Represents transaction category

4. Customer Information
   - Age (0.039042)
   - Gender (0.010257)

## Technical Implementation

### Model Architecture

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

### Key Components

1. **Data Preprocessing**:

   - StandardScaler for feature normalization
   - Label encoding for categorical variables
   - Date/time processing for temporal features

2. **Class Imbalance Handling**:

   - SMOTE with 0.3 sampling strategy
   - Preserves data distribution while addressing imbalance

3. **Validation Strategy**:
   - 5-fold cross-validation
   - Stratified sampling
   - F1 score as primary metric

## Features

Six core features were used:

- `amt`: Transaction amount
- `trans_hour`: Hour of transaction
- `trans_day`: Day of transaction
- `category_code`: Encoded transaction category
- `age`: Customer age
- `gender`: Customer gender (encoded)

## Model Development Process

### Feature Engineering

1. **Temporal Features**:

   - Extracted hour and day from transaction datetime
   - Preserved temporal patterns in transactions

2. **Categorical Encoding**:

   - Label encoded transaction categories
   - Binary encoded gender

3. **Age Calculation**:
   - Precise age calculation using transaction date
   - Considered month and day for accuracy

### Model Configuration

- Random Forest with 220 trees
- SMOTE for balanced training
- Parallel processing enabled
- Cross-validation for robust evaluation

## Results Analysis

### Model Stability

- Very stable performance across folds
- Low standard deviation (±0.0017)
- Consistent F1 scores around 0.974

### Prediction Distribution

- Realistic distribution in test set
- Maintains expected fraud ratio
- Aligns with real-world fraud patterns
