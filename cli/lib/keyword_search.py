import string
import os
import pickle
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT, 
    load_movies, 
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m['id']
            text = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:     
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f: 
            pickle.dump(self.docmap, f)
        with open(self.tf_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if not (os.path.exists(self.index_path) 
                and os.path.exists(self.docmap_path) 
                and os.path.exists(self.tf_path)):
            raise FileNotFoundError
        with open(self.index_path, 'rb') as f:     
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f: 
            self.docmap = pickle.load(f)
        with open(self.tf_path, 'rb') as f: 
            self.term_frequencies = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
        
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
    
    def get_tf(self, doc_id: int, term: str) -> int:
        term = tokenize_text(term)
        if len(term) != 1:
            raise ValueError("term must be a single token")
        term = term[0]
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("index and docmap files are not found")
        return
    preprocessed_query = tokenize_text(query)
    results = []
    seen = set()
    for token in  preprocessed_query:
        matching_doc_ids = index.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = index.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for filtered_word in filtered_words:
        stemmed_words.append(stemmer.stem(filtered_word))
    return stemmed_words

def tf_command(doc_id: int, term: str):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("index and docmap and term_frequences files are not found")
        return
    return index.get_tf(doc_id, term)

def idf_command(term: str):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("index and docmap and term_frequences files are not found")
        return
    return index.get_idf(term)

def tfidf_command(doc_id: int, term: str):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("index and docmap and term_frequences files are not found")
        return
    return index.get_tf_idf(doc_id, term)
