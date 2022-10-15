import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




def tokenization(text: str):
    return pd.Series(nltk.word_tokenize(text.lower()))


def removal(tokens: pd.Series):

    stopwords_list = stopwords.words("english")

    tokens = tokens.apply(lambda token: token.translate(str.maketrans('', '', string.punctuation)))
    tokens = tokens.apply(lambda token: token if token not in stopwords_list and token != '' else None).dropna()

    return tokens


def stemming(tokens: pd.Series):

    stemmer = PorterStemmer()

    return tokens.apply(lambda token: stemmer.stem(token))


def lemmatization(tokens: pd.Series):

    lemmatizer = WordNetLemmatizer()

    return tokens.apply(lambda token: lemmatizer.lemmatize(token))


def query_expansion(tokens: pd.Series, sample_size=2):

    token_list = tokens.tolist()

    new_tokenlist = []
    for token in token_list:
        synonyms = get_synonyms(token, sample_size)

        new_tokenlist.append(token)
        if len(synonyms) > 0:
            new_tokenlist.extend(synonyms)

    return pd.Series(new_tokenlist)


def get_synonyms(phrase, sample_size):

    synonyms = []
    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            if '_' not in l.name() and l.name() != phrase:
                synonyms.append(l.name())

    synonym_set = set(synonyms)

    if sample_size > len(synonym_set):
        return list(synonym_set)
    else:
        synonym_set_sampled = random.sample(synonym_set, sample_size)
        return list(synonym_set_sampled)
