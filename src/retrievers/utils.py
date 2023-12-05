from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def tokenize_and_stem(text):
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords (optional, depending on your needs)
    words = [word for word in words if word.lower() not in stopwords.words("english")]

    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words
