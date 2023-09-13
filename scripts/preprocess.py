import os
import re

import nltk
import pandas as pd
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


def preprocess_data(input_file, output_file, chunk_size=50000):
    download_nltk_data()

    stop_words = set(stopwords.words('english'))
    custom_stopwords = [
         "ability", "able", "absolute", "absolutely", "account", "accurate", "achieve", "address",
         "allowing", "also", "analyze", "analyzes", "answer", "application", "approach",
         "around", "article", "aspect", "audience", "author", "available", "based", "begin", "best", "better",
         "beyond", "bound", "brief", "called", "capable", "capture", "carefully", "case", "certain", "challenging",
         "compare", "complex", "component", "comprehensive","concept", "conceptual",
         "conclusion", "condition", "conduct", "conjecture", "consider", "construct", "content", "context", "cost",
         "cross", "crucial", "current", "demonstarte", "demonstrate", "derive", "derived", "describe",
         "described", "describes", "detailed", "determine", "developed", "different", "difficult", "directly",
         "discourse", "discuss", "distinguish", "driven", "due", "effect", "effective", "efficient", "efficiently",
         "eight", "element", "emphasis", "end", "evaluate", "even", "example", "experiment", "experimental",
         "explain", "extensive", "family", "feature", "figuer", "figure", "find", "fine", "finite", "finitely", "first",
         "fit", "five", "found", "four", "framework", "function", "fundamental", "future", "general", "give",
         "given", "good", "grained", "graph", "group", "handed", "high", "higher", "however", "illustrate", "impact",
         "implement", "important", "include", "integrate", "interest", "introduce", "introduced", "introduction",
         "investigate", "issue", "iteration", "known", "large", "last", "left", "let", "long", "low", "lower", "make",
         "many", "maximal", "may", "methdo", "method", "methodology", "minimal", "model", "moreover", "multiple",
         "necessary", "need", "needed", "new", "news", "nine", "non", "note", "novel", "number", "numerical",
         "objective", "observables", "observation", 'obtained', "often", "one", "open", "operator", "optimal", "order", "outline",
         "outlines", "output", "paper", "papr", "parameter", "parametre", "part", "particular", "perform",
         "performance", "performed", "performing", "performnace", "phase", "point", "possible",
         "potential", "pre", "precisely", "present", "previous", "principle", "problem", "problme", "process", "prof",
         "proof", "proper", "property", "propose", "proposed", "proposes", "propsoed", "prove", "provide", "provided",
         "publicly", "publish", "purpose", "quality", "question", "range", "real", "recent", "recently",
         "recommendation", "related", "representation", "require", "research", "result", "rev", "review",
         "right", "rigorous", "role", "scale", "scenario", "second", "section", "series", "serious", "set", "setting",
         "seven", "show", "shwo", "significant", "significantly", "simulation", "single", "six", "solution", "state",
         "strongly", "structure", "studied", "study", "sufficient", "suggestion", "sum", "synthesize", "system",
         "table", "take", "taken", "task", "technique", "ten", "term", "theorem", "theory", "third",
         "though", "three", "time", "topic", "two", "type", "upper", "use", "used", "using", "utilize", "valid",
         "value", "variable", "variety", "various", "via", "view", "way", "well", "whether", "wide", "widely", "within",
         "without", "work", "world", "written", "year", "zero", "zeroth"]
    stop_words.update(custom_stopwords)
    lemmatizer = WordNetLemmatizer()

    with open(input_file, 'r') as f:
        total_rows = sum(1 for line in f)

    output_exists = os.path.isfile(output_file)

    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
            chunk = chunk[['title', 'abstract', 'categories', 'update_date']].drop_duplicates(
                subset=['title', 'abstract'])

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
