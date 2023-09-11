import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def download_nltk_data():
    """
    Ensure necessary NLTK data is downloaded.
    """
    datasets = ["stopwords", "wordnet", "punkt"]
    for dataset in datasets:
        try:
            nltk.data.find(f'corpora/{dataset}')
        except LookupError:
            nltk.download(dataset)


def preprocess_text(text):
    """
    Enhanced preprocessing of the text data.
    """
    if pd.isna(text):
        return ''

    #regex pattern for cleaning
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # Remove non-alphanumeric characters

    
    #tokenization
    words = word_tokenize(text.lower())

    # Add custom stopwords
    custom_stopwords = ['use', 'study', 'result', 'show', 'doi' 
                        'paper', 'author', 'figure', 'table', 'data',
                        'approach', 'technique','case',
                        'provide', 'based', 'work', 'present', 'also',
                        'method', 'model', 'performance', 'problem',
                        'proposed', 'using']
    stop_words.update(custom_stopwords)

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)


def preprocess_data(input_file, output_file, chunk_size=20000):
    """
    Main preprocessing function.
    """
    # Download necessary NLTK data
    download_nltk_data()

    # Use context manager for file operations
    with open(input_file, 'r') as f:
        total_rows = sum(1 for line in f)

    output_exists = os.path.isfile(output_file)
    
    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
            chunk = chunk[['title', 'abstract', 'categories', 'update_date']]
            
            # Remove repeated rows
            chunk = chunk.drop_duplicates(subset=['title', 'abstract'])

            # Vectorized text preprocessing
            for column in ['title', 'abstract']:
                chunk[column] = chunk[column].apply(preprocess_text)

            # Concatenate title and abstract columns
            chunk['text'] = chunk['title'] + ' ' + chunk['abstract']
            chunk = chunk.drop(['title', 'abstract'], axis=1)

            # Remove rows with empty text
            chunk = chunk[chunk['text'] != '']

            # Convert update_date to datetime
            chunk['update_date'] = pd.to_datetime(chunk['update_date'])

            # Write to CSV
            if not output_exists:
                chunk.to_csv(output_file, mode='w', index=False)
                output_exists = True  # Set the flag to True after first write
            else:
                chunk.to_csv(output_file, mode='a', index=False, header=False)
            
            pbar.update(chunk_size)


if __name__ == "__main__":
    # Initialize stopwords
    stop_words = set(stopwords.words('english'))
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    raw_path = '../data/arxiv-metadata-oai.json'
    processed_path = '../data/data_preprocessed.csv'
    preprocess_data(raw_path, processed_path)
