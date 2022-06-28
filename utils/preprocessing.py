from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from utils.pdf_parsing import pdf_to_text
from typing import List


def preprocess_document(document: str) -> List[str]:
    stemmer = SnowballStemmer(language='english')
    stop_words = stopwords.words('english')
    number_token_name = "special_number_token"

    # Tokenize, stem, remove special characters and stopwords.
    tokens = word_tokenize(document)
    tokens = [t for t in tokens if t.isalnum() and len(t) > 1 and t not in stop_words]
    tokens = [t.lower() for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [number_token_name if t.isnumeric() else t for t in tokens]
    return tokens


def preprocess_pdf_document(pdf_path: str) -> List[str]:
    # Parse pdf
    parsed_text = pdf_to_text(pdf_path)
    # Preprocess text
    preprocessed_document = preprocess_document(parsed_text)
    return preprocessed_document
