import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
nlp = spacy.load("en_core_web_sm")
text = "Hello world! This is an example sentence to demonstrate text processing using spaCy."
doc = nlp(text)
tokens = [token.text for token in doc]
print("Tokens:", tokens)
lemmas = [token.lemma_ for token in doc]
print("Lemmas:", lemmas)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stems = [stemmer.stem(token.text) for token in doc if not token.is_punct]
print("Stems:", stems)
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("Filtered Tokens (Stop Words Removed):", filtered_tokens)
punctuation_removed = [token.text for token in doc if not token.is_punct]
print("Tokens with Punctuation Removed:", punctuation_removed)

# Complete processing and print results
print("\nProcessed Text:")
print("Original Text:", text)
print("Tokens:", tokens)
print("Lemmas:", lemmas)
print("Stems:", stems)
print("Filtered Tokens (Stop Words Removed):", filtered_tokens)
print("Tokens with Punctuation Removed:", punctuation_removed)

    
