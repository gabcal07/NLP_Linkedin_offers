import sys
import os

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from models.src.preprocessing.tokenizer import tokenize, BYTE_TOKENIZER

# Download the NLTK data files (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class TextNormalizer:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.inflect_engine = inflect.engine()

    def _convert_numbers_to_words(self, text):
        # Convert numbers to words
        words = text.split()
        for i, word in enumerate(words):
            if word.isdigit():
                words[i] = self.inflect_engine.number_to_words(word)
        return " ".join(words)

    def normalize_text_for_topic_modeling(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize the text
        tokens = tokenize(text, method="byte")

        # Clean the Ġ tokens
        tokens = [token.replace("Ġ", "") for token in tokens]

        # Remove stop words
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatize the words
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # tokens = [self.stemmer.stem(word) for word in tokens]

        # Convert numbers to words
        tokens = [
            self._convert_numbers_to_words(word) if word.isdigit() else word
            for word in tokens
        ]

        normalized_text = " ".join(tokens)

        return normalized_text
