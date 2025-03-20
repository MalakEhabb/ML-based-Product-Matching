## README: Product Matching Model

## Overview
This repository contains a product matching model for product names, designed to standardize and match seller item names to marketplace names. It includes preprocessing, training, evaluation, and reusability features.

## Preprocessing
- Arabic text normalization (removal of diacritics, standardizing characters like `أ`, `إ`, `آ`, etc.).
- Stopword removal (e.g., "جديد", "قديم", "سعر").
- Cleaning seller item names and marketplace product names.

## Vectorization
- Using `TfidfVectorizer` with character n-grams (2-3 grams) and a maximum of 3000 features.
- Converts cleaned text data into numerical vectors for machine learning.

## Model Training
- Train-test split (80-20 split).
- Model: `RandomForestClassifier` with 50 estimators and a fixed random seed.
- Model training using the transformed data.

## Model Evaluation
- Accuracy score based on sku matching (93%).
- Cross-validation score with 5 folds.

## Model Saving 
- The trained model and vectorizer are saved using `joblib`.

## Visualization
- A bar chart visualizes false matches categorized by confidence levels.
- A pie chart shows the percentage of false matches above and below the threshold.

## Walkthrough for model reusage
- Download the repository
- Unzip the product_matching_model and the vectorizer 
- install the requirments.txt
- run the predict.py script
- check the directory for the resulted excel sheet

## Example usage 
```bash
python predict.py <file.xlsx> <MasterSheet> <DatasetSheet>
```
