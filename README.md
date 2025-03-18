## README: Product Matching Model

## Overview
This repository contains a product matching model for Arabic product names, designed to standardize and match seller item names to marketplace names. It includes preprocessing, training, evaluation, and reusability features.

## Preprocessing
- Arabic text normalization (removal of diacritics, standardizing characters like `أ`, `إ`, `آ`, etc.).
- Stopword removal (e.g., "جديد", "قديم", "سعر").
- Cleaning seller item names and marketplace product names.

## Vectorization
- Using `TfidfVectorizer` with character n-grams (2-3 grams) and a maximum of 6000 features.
- Converts cleaned text data into numerical vectors for machine learning.

## Model Training
- Train-test split (80-20 split).
- Model: `RandomForestClassifier` with 100 estimators and a fixed random seed.
- Model training using the transformed data.

## Model Evaluation
- Accuracy score on the test set.
- Cross-validation score with 5 folds.

## Model Saving and Reusage
- The trained model and vectorizer are saved using `joblib`.
- A pipeline function allows loading the model and applying it to new datasets.
- Predictions are made, and confidence scores are generated.
- Predicted product matches are assigned SKUs from a master file.

## Visualization
- A bar chart visualizes false matches categorized by confidence levels.
- A pie chart shows the percentage of false matches above and below a threshold.



