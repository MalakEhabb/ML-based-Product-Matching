import pandas as pd
import numpy as np
import joblib
import re
import time
import sys
import os

def normalize_arabic(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove tashkeel
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ي", "ى")
    return text

def clean_corpus(corpus, words_to_remove):
    cleaned_corpus = []
    for text in corpus:
        if isinstance(text, str):
            text = normalize_arabic(text)
            for word in words_to_remove:
                word = normalize_arabic(word)
                pattern = r'\b' + re.sub(r'(.)\1*', r'\1+', re.escape(word)) + r'\b'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.UNICODE)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = ""
        cleaned_corpus.append(text)
    return cleaned_corpus

# Load model and vectorizer
extract_path = "product_matching_model"
model_path = r"product_matching_model/product_matching_model.pkl"
vectorizer_path = r"vectorizer.pkl"

print("🔹 Loading Model & Vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def product_matching_pipeline(excel_file_path, masterfile_sheet, dataset_sheet, words_to_remove):
    print("🔹 Loading Excel File...")
    try:
        masterfile = pd.read_excel(excel_file_path, sheet_name=masterfile_sheet)
        dataset = pd.read_excel(excel_file_path, sheet_name=dataset_sheet)
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

    required_master_cols = {'product_name_ar', 'sku'}
    required_dataset_cols = {'seller_item_name', 'marketplace_product_name_ar'}

    if not required_master_cols.issubset(masterfile.columns):
        print(f"❌ Missing columns in Master File: {required_master_cols - set(masterfile.columns)}")
        return None
    if not required_dataset_cols.issubset(dataset.columns):
        print(f"❌ Missing columns in Dataset: {required_dataset_cols - set(dataset.columns)}")
        return None

    print("🔹 Cleaning Text Data...")
    masterfile['marketplace_name_clean'] = clean_corpus(masterfile['product_name_ar'].astype(str), words_to_remove)
    dataset['seller_item_name_clean'] = clean_corpus(dataset['seller_item_name'].astype(str), words_to_remove)
    dataset['marketplace_name_clean'] = clean_corpus(dataset['marketplace_product_name_ar'].astype(str), words_to_remove)

    print("🔹 Transforming Data...")
    X_dataset = vectorizer.transform(dataset['seller_item_name_clean'])

    print("🔹 Predicting Matches...")
    y_pred_dataset = model.predict(X_dataset)
    y_pred_proba = model.predict_proba(X_dataset)
    confidence_scores = np.max(y_pred_proba, axis=1)

    dataset['predicted_marketplace_name'] = y_pred_dataset
    dataset['confidence_score'] = confidence_scores

    sku_map = masterfile.set_index('marketplace_name_clean')['sku'].to_dict()
    dataset['matched_sku'] = dataset['predicted_marketplace_name'].apply(lambda x: sku_map.get(x, 'Not Found'))

    dataset = dataset.drop(columns=['seller_item_name_clean', 'marketplace_name_clean', 'predicted_marketplace_name'])
    return dataset

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <file.xlsx> <MasterSheet> <DatasetSheet>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    master_sheet = sys.argv[2]
    dataset_sheet = sys.argv[3]
    output_file = "final_matched_dataset.xlsx"
    
    words_to_remove = ['شريط', 'جديد', 'قديم', 'سعر', 'سانوفي', 'افنتس', 'ابيكو', 'ج', 'س', 
        'العامرية', 'كبير', 'صغير', 'هام', 'مهم', 'احذر', 'يوتوبيا', 'دوا', 
        'ادويا', 'لا يرتجع', 'يرتجع', 'عادي', 'ميباكو']
    
    print("🚀 Starting Product Matching Process...")
    start_time = time.time()
    
    final_dataset = product_matching_pipeline(input_file, master_sheet, dataset_sheet, words_to_remove)
    
    end_time = time.time()
    
    if final_dataset is not None:
        final_dataset.to_excel(output_file, index=False)
        print(f"✅ Processing completed in {end_time - start_time:.2f} seconds.")
        print(f"📂 Results saved in {output_file}")
    else:
        print("❌ Error: Could not process the dataset.")
