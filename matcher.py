import re
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_arabic(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove diacritics
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
    
    df = pd.read_excel(file_path, sheet_name=dataset_sheet)
    df['seller_item_name_clean'] = clean_corpus(df['seller_item_name'].astype(str), words_to_remove)
    df['marketplace_name_clean'] = clean_corpus(df['marketplace_product_name_ar'].astype(str), words_to_remove)
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=6000)
    X = vectorizer.fit_transform(df['seller_item_name_clean'])
    
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save vectorizer for reuse
    df.to_csv('cleaned_dataset.csv', index=False)
    print("Processing complete. Cleaned data saved as 'cleaned_dataset.csv'")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python matcher.py <file.xlsx> <MasterSheet> <DatasetSheet>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    master_sheet = sys.argv[2]
    dataset_sheet = sys.argv[3]
    
    process_excel(file_path, master_sheet, dataset_sheet)
