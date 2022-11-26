## for data
import json
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse

## for processing
import re
import nltk

## for bag-of-words
from sklearn import feature_extraction, feature_selection, model_selection, naive_bayes, pipeline, manifold, \
    preprocessing, __all__

## for explainer
from lime import lime_text

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for deep learning
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, \
    precision_recall_curve, f1_score
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for bert language model
import transformers


def feature_generation(X_training, y_training, n_gram, tf_idf=True):
    """
    :param X_training: training data
    :param y_training: target data
    :param n_gram: set <integer> for bigram, trigram etc.
    :param tf_idf: If True, return tfidf features, otherwise return bow features
    :return: sparse matrix, re-fitted sparse matrix (chi2), tf/tfidf features and vectorized training data
    """

    # Feature Engineering (vocabulary size = 10.000 words)
    def check_vectorization():
        if tf_idf:
            ## TF-IDF (advanced variant of BoW)
            TFIDF = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, n_gram))
            return TFIDF
        else:
            ## Count (classic BoW)
            BOW = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1, n_gram))
            return BOW

    vectorizer = check_vectorization()

    ##Extract Vocabulary
    corpus = X_training["preprocessed_text"].values.astype(str)
    vectorizer.fit(corpus)
    X_training = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_
    print("Training vocabulary size before dimension reduction: " + str(len(dic_vocabulary)))

    ##Create Sparse Matrix: 6072 (No. of Docs. or rows in training set) x 10000 (vocabulary size)
    sns.set()
    sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample (non-zero values in black)')
    plt.savefig("img/" + "sparse_matrix.png")
    plt.ioff()
    plt.show()

    ##Look up Position of a certain word in the Sparse Matrix
    word = "think"
    print("Position of the word " + word + " in matrix: " + str(dic_vocabulary[word]))

    # Feature Selection
    ##Reduce Dimensionality for sparse data with Chi-Quadrat (further see)
    X_names = vectorizer.get_feature_names_out()
    p_value_limit = 0.95
    features = pd.DataFrame()
    print("Top Features for Each Class:")
    for cat in np.unique(y_training):
        chi2, p = feature_selection.chi2(X_training, y_training == cat)
        features = features.append(pd.DataFrame(
            {"feature": X_names, "score": 1 - p, "y": cat}))
        features = features.sort_values(["y", "score"], ascending=[True, False])
        features = features[features["score"] > p_value_limit]
    X_names = features["feature"].unique().tolist()

    for cat in np.unique(y_training):
        print("# {}:".format(cat))
        print("  . selected features:",
              len(features[features["y"] == cat]))
        print("  . top features:", ",".join(
            features[features["y"] == cat]["feature"].values[:10]))
        print(" ")

    ##Re-Fit vectorizer on corpus with new set of words and create new sparse matrix
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    vectorizer.fit(corpus)
    X_training = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_
    print("Training vocabulary size after dimension reduction: " + str(len(dic_vocabulary)))

    # New Sparse Matrix
    sns.set()
    sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample (non-zero values in black)')
    plt.savefig("img/" + "re_fit_sparse_matrix.png")
    plt.ioff()
    plt.show()

    return X_training, vectorizer


##Classification
def naiveBayes(X_training, y_training, X_testing, y_testing, vectorizer):
    classifier = naive_bayes.MultinomialNB()

    ## pipeline

    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])

    ## train classifier
    model["classifier"].fit(X_training, y_training)

    ## test
    X = X_testing["preprocessed_text"].values.astype(str)
    predicted = model.predict(X)
    predicted_prob = model.predict_proba(X)

    ## Evaluation
    classes = np.unique(y_testing)
    y_test_array = pd.get_dummies(y_testing, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_testing, predicted)
    auc = roc_auc_score(y_testing, predicted_prob, multi_class="ovr")
    micro_f1 = f1_score(y_testing, predicted, average="micro")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Micro F1:", round(micro_f1, 2))
    print()
    print("Detail:")
    print(classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------

# Initialize

# Load dataset
df = pd.read_csv("data/df_multi_preprocessed.csv")

# Create training and test split
## get X
X_train, X_test = model_selection.train_test_split(df, test_size=0.3, random_state=42)

## get target
y_train = X_train["type"].values
y_test = X_test["type"].values

# Call Functions

# Feature Generation and Vectorizing of Textdata
training, tfidf = feature_generation(X_train, y_train, 2, tf_idf=True)
#training, bow = feature_generation(X_train, y_train, 2, tfidf=False)

# Classification
naiveBayes(training, y_train, X_test, y_test, tfidf)
