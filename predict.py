import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_arabic(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove tashkeel
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")  # Normalize أ
    text = text.replace("ة", "ه")  # Normalize ة
    text = text.replace("ي", "ى")  # Normalize ي
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

def match_products(input_excel):
    print("Loading model and vectorizer...")
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    print(f"Reading input file: {input_excel}")
    df = pd.read_excel(input_excel)

    if 'seller_item_name' not in df.columns:
        raise ValueError("Input Excel must contain 'seller_item_name' column")
    
    print("Cleaning text...")
    words_to_remove = ['شريط', 'جديد', 'قديم', 'سعر', 'سانوفي', 'افنتس', 'ابيكو', 'ج', 'س',
                       'العامرية', 'كبير', 'صغير', 'هام', 'مهم', 'احذر', 'يوتوبيا', 'دوا',
                       'ادويا', 'لا يرتجع', 'يرتجع', 'عادي', 'ميباكو']
    df['seller_item_name_clean'] = clean_corpus(df['seller_item_name'].astype(str), words_to_remove)
    
    print("Transforming text...")
    X = vectorizer.transform(df['seller_item_name_clean'])
    
    print("Predicting matches...")
    df['predicted_marketplace_name'] = model.predict(X)
    
    output_excel = input_excel.replace('.xlsx', '_matched.xlsx')
    print(f"Saving results to {output_excel}")
    df[['seller_item_name', 'predicted_marketplace_name']].to_excel(output_excel, index=False)

    print("Matching complete.")

