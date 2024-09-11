import nltk
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample text for processing
text = "Hello world! This is an example sentence to demonstrate text processing using NLTK."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmas:", lemmas)

# Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(token) for token in tokens]
print("Stems:", stems)

# Stop Word Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("Filtered Tokens (Stop Words Removed):", filtered_tokens)

# Punctuation Removal
punctuation_table = str.maketrans('', '', string.punctuation)
punctuation_removed = [token.translate(punctuation_table) for token in tokens if token not in string.punctuation]
print("Tokens with Punctuation Removed:", punctuation_removed)

# Complete processing and print results
print("\nProcessed Text:")
print("Original Text:", text)
print("Tokens:", tokens)
print("Lemmas:", lemmas)
print("Stems:", stems)
print("Filtered Tokens (Stop Words Removed):", filtered_tokens)
print("Tokens with Punctuation Removed:", punctuation_removed)
