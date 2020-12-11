"""
API Reference :  https://www.nltk.org/api/nltk.stem.html
"""

from nltk.stem import PorterStemmer, WordNetLemmatizer

def porter_stemmer(token_list):
    stemmer = PorterStemmer()

    stemmed_tokens = []

    for token in token_list:
        stemmed_tokens.append(stemmer.stem(token))

    return list(set(stemmed_tokens))


def wordnet_lemmatizer(token_list):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []
    
    for token in token_list:
        lemmatized_tokens.appen(lemmatizer.lemmatize(token))

    return list(set(lemmatized_tokens))


