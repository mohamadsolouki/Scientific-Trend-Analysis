import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def download_nltk_data():
    datasets = ["stopwords", "wordnet", "punkt"]
    for dataset in datasets:
        try:
            nltk.data.find(f'corpora/{dataset}')
        except LookupError:
            nltk.download(dataset)

def preprocess_text(text, stop_words, lemmatizer):
    if pd.isna(text):
        return ''
    
    # Cleaning operations
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)

    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

def preprocess_data(input_file, output_file, chunk_size=20000):
    download_nltk_data()

    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['use', 'study', 'result', 'show', 'paper', 'figure', 'via', 
                        'table', 'approach', 'technique','case', 'result', 'provide', 
                        'based', 'work', 'present', 'method', 'model', 'performance', 
                        'problem', 'proposed', 'using', 'used', 'new', 'one', 'two', 
                        'time', 'may', 'number', 'first', 'set', 'state', 'many', 'well',
                        'different', 'general', 'found', 'property', 'derived', 'also',
                        'find', 'known', 'given', 'related', 'provided', 'make', 'group',
                        'called', 'certain', 'lower', 'higher', 'bound', 'setting', 'moreover']
    stop_words.update(custom_stopwords)
    lemmatizer = WordNetLemmatizer()

    with open(input_file, 'r') as f:
        total_rows = sum(1 for line in f)

    output_exists = os.path.isfile(output_file)
    
    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
            chunk = chunk[['title', 'abstract', 'categories', 'update_date']].drop_duplicates(subset=['title', 'abstract'])
            
            chunk['title'] = chunk['title'].apply(preprocess_text, args=(stop_words, lemmatizer))
            chunk['abstract'] = chunk['abstract'].apply(preprocess_text, args=(stop_words, lemmatizer))
            
            chunk['text'] = chunk['title'] + ' ' + chunk['abstract']
            chunk = chunk.drop(['title', 'abstract'], axis=1)
            chunk = chunk[chunk['text'] != '']

            chunk['update_date'] = pd.to_datetime(chunk['update_date'])

            if not output_exists:
                chunk.to_csv(output_file, mode='w', index=False)
                output_exists = True
            else:
                chunk.to_csv(output_file, mode='a', index=False, header=False)
            
            pbar.update(chunk_size)

if __name__ == "__main__":
    raw_path = '../data/arxiv-metadata-oai.json'
    processed_path = '../data/data_preprocessed.csv'
    preprocess_data(raw_path, processed_path)
