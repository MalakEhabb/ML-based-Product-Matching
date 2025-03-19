import re
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

def normalize_arabic(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove tashkeel
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")  # Normalize Alef
    text = text.replace("ة", "ه")  # Normalize Ta Marbuta
    text = text.replace("ي", "ى")  # Normalize Ya
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
        cleaned_corpus.append(text)
    return cleaned_corpus

def process_excel(file_path, master_sheet, dataset_sheet):
    words_to_remove = ['شريط', 'جديد', 'قديم', 'سعر', 'سانوفي', 'افنتس', 'ابيكو', 'ج', 'س',
                       'العامرية', 'كبير', 'صغير', 'هام', 'مهم', 'احذر', 'يوتوبيا', 'دوا',
                       'ادويا', 'لا يرتجع', 'يرتجع', 'عادي', 'ميباكو']
    
    df_master = pd.read_excel(file_path, sheet_name=master_sheet)
    df_dataset = pd.read_excel(file_path, sheet_name=dataset_sheet)
    
    df_dataset['seller_item_name_clean'] = clean_corpus(df_dataset['seller_item_name'].astype(str), words_to_remove)
    df_master['marketplace_name_clean'] = clean_corpus(df_master['marketplace_product_name_ar'].astype(str), words_to_remove)
    
    # Load pre-trained vectorizer and model from the same directory
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("encoder.pkl")
    model = load_model("model.h5")
    
    # Transform input data
    X = vectorizer.transform(df_dataset['seller_item_name_clean']).toarray()
    
    processing_times = []
    matched_skus = []
    
    # Predict
    for index, row in df_dataset.iterrows():
        start_time = time.time()
        pred_vector = vectorizer.transform([row['seller_item_name_clean']]).toarray()
        y_pred = np.argmax(model.predict(pred_vector), axis=1)
        predicted_name = encoder.inverse_transform(y_pred)[0]
        df_dataset.at[index, 'predicted_category'] = predicted_name
        
        # Find corresponding SKU
        matched_row = df_master[df_master['marketplace_name_clean'] == predicted_name]
        sku = matched_row['sku'].values[0] if not matched_row.empty else "Not Found"
        df_dataset.at[index, 'matched_sku'] = sku
        
        end_time = time.time()
        processing_times.append(end_time - start_time)
        matched_skus.append(sku)
    
    df_dataset['processing_time'] = processing_times
    df_dataset.to_csv('matched_results.csv', index=False)
    print("Processing complete. Results saved as 'matched_results.csv'")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python matcher.py <file.xlsx> <MasterSheet> <DatasetSheet>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    master_sheet = sys.argv[2]
    dataset_sheet = sys.argv[3]
    
    process_excel(file_path, master_sheet, dataset_sheet)
