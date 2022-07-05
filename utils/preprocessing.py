from nltk import SnowballStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from utils.pdf_parsing import pdf_to_text
from typing import List


def preprocess_document(document: str) -> List[str]:
    stemmer = SnowballStemmer(language='english')
    stop_words = stopwords.words('english')

    # Tokenize, stem, remove special characters and stopwords.
    tokens = word_tokenize(document)
    tokens = [t for t in tokens if t.isalnum() and len(t) > 1 and t not in stop_words]
    tokens = [t.lower() for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if not t.isnumeric()]
    return tokens


def preprocess_pdf_document(pdf_path: str) -> List[str]:
    # Parse pdf
    parsed_text = pdf_to_text(pdf_path)
    # Preprocess text
    preprocessed_document = preprocess_document(parsed_text)
    return preprocessed_document


def clean_document_for_embedding(pdf_path):
    document = pdf_to_text(pdf_path)
    sentences = sent_tokenize(document)
    result = []
    for s in sentences:
        cleaned = s.replace("-\n", "")
        cleaned = cleaned.replace("\n", " ")
        cleaned = cleaned.replace("\x0c", "")
        cleaned = cleaned.replace("\t", " ")
        cleaned = cleaned.replace("  ", " ")
        result.append(cleaned)
    return result


def clean_query_for_embedding(query):
    cleaned = query.replace("-\n", "")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("\x0c", "")
    cleaned = cleaned.replace("\t", " ")
    cleaned = cleaned.replace("  ", " ")
    return [cleaned]
