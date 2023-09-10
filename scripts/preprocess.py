import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize global objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ''
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert text to lowercase and tokenize
    words = text.lower().split()
    
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def main():
    input_file = '../data/arxiv-metadata-oai.json'
    output_file = '../data/data_preprocessed.csv'
    chunk_size = 20000
    
    # Get the total number of rows in the JSON file
    total_rows = sum(1 for line in open(input_file))
    
    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
            # Keep only the necessary columns
            chunk = chunk[['id', 'title', 'abstract', 'categories', 'update_date']]
            
            # Preprocess the 'title' and 'abstract' columns individually
            chunk['title'] = chunk['title'].apply(preprocess_text)
            chunk['abstract'] = chunk['abstract'].apply(preprocess_text)
            
            # Convert the 'update_date' column to datetime
            chunk['update_date'] = pd.to_datetime(chunk['update_date'])
            
            # Keep only the latest version of each article
            chunk = chunk.sort_values('update_date').drop_duplicates(subset=['title', 'abstract'], keep='last')
            
            # Write to the CSV file
            mode = 'a' if pbar.n > 0 else 'w'  # Append mode for all chunks except the first
            chunk.to_csv(output_file, mode=mode, header=(pbar.n == 0), index=False)
            
            pbar.update(chunk_size)

if __name__ == '__main__':
    main()
