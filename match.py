import pandas as pd
import numpy as np
import joblib
import re
import time
import sys
from sklearn.preprocessing import normalize

def normalize_arabic(text):
    """
    Normalize Arabic text by removing tashkeel and standardizing characters.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\u064B-\u065F]', '', text)   
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")   
    text = text.replace("ة", "ه")  
    text = text.replace("ي", "ى")  
    return text

def clean_corpus(corpus, words_to_remove):
    """
    Clean a corpus of text by normalizing and removing specified words.
    """
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

def product_matching_pipeline(excel_file_path, masterfile_sheet, dataset_sheet, words_to_remove):
    """
    Run the product matching pipeline.
    """
    print("🔹 Loading Excel File...")
    try:
        
        masterfile = pd.read_excel(excel_file_path, sheet_name=masterfile_sheet)
        dataset = pd.read_excel(excel_file_path, sheet_name=dataset_sheet)
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

    # Check for required columns
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

    print("🔹 Loading Model and Vectorizer...")
    try: 
        model = joblib.load("product_matching_model/product_matching_model.pkl")
        vectorizer = joblib.load("vectorizer/vectorizer.pkl")
    except Exception as e:
        print(f"❌ Error loading model or vectorizer: {e}")
        return None

    print("🔹 Transforming Data...") 
    X_dataset = vectorizer.transform(dataset['seller_item_name_clean'])

    print("🔹 Predicting Matches...") 
    y_pred_proba = model.predict_proba(X_dataset)   
 
    temperature = 0.5   
    y_pred_proba_scaled = np.power(y_pred_proba, 1 / temperature)  
    y_pred_proba_scaled = normalize(y_pred_proba_scaled, norm='l1', axis=1)   
 
    y_pred_dataset = np.argmax(y_pred_proba_scaled, axis=1)   
    confidence_scores = np.max(y_pred_proba_scaled, axis=1)  
 
    class_labels = model.classes_  
    dataset['predicted_marketplace_name'] = [class_labels[idx] for idx in y_pred_dataset]
 
    dataset['confidence_score'] = confidence_scores
 
    sku_map = masterfile.set_index('marketplace_name_clean')['sku'].to_dict()
 
    dataset['matched_sku'] = dataset['predicted_marketplace_name'].apply(lambda x: sku_map.get(x, 'Not Found'))
 
    dataset = dataset.drop(columns=['seller_item_name_clean', 'marketplace_name_clean', 'predicted_marketplace_name'])

    return dataset

def main():
    """
    Main function to run the product matching pipeline.
    """
    if len(sys.argv) != 4:
        print("Usage: python match.py <file.xlsx> <MasterSheet> <DatasetSheet>")
        sys.exit(1)
 
    excel_file_path = sys.argv[1]
    masterfile_sheet = sys.argv[2]
    dataset_sheet = sys.argv[3]

    # Define words to remove
    words_to_remove = ['شريط', 'جديد', 'قديم', 'سعر', 'سانوفي', 'افنتس', 'ابيكو', 'ج', 'س', 
                       'العامرية', 'كبير', 'صغير', 'هام', 'مهم', 'احذر', 'يوتوبيا', 'دوا', 
                       'ادويا', 'لا يرتجع', 'يرتجع', 'عادي', 'ميباكو']

    print("🚀 Starting Product Matching Process...")
    start_time = time.time()

    # Run the product matching pipeline
    final_dataset = product_matching_pipeline(excel_file_path, masterfile_sheet, dataset_sheet, words_to_remove)

    end_time = time.time()

    # Save the results if the pipeline was successful
    if final_dataset is not None:
        output_file = "final_matched_dataset.xlsx"
        final_dataset.to_excel(output_file, index=False)
        print(f"✅ Processing completed in {end_time - start_time:.2f} seconds.")
        print(f"📂 Results saved in {output_file}")
    else:
        print("❌ Error: Could not process the dataset.")

if __name__ == "__main__":
    main()
