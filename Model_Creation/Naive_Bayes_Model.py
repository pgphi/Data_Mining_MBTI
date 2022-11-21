# install/import packages
import pandas as pd
import numpy as np
import os

import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # good for continous feautures
from sklearn.naive_bayes import BernoulliNB  # suitable for discrete feautures; especially for boolean
from sklearn.naive_bayes import MultinomialNB  # good for discrete feautures ( also fractional counts i.e. TFIDF)
from sklearn.naive_bayes import CategoricalNB  # good for discrete feautures and catagorically distributed
from sklearn.naive_bayes import ComplementNB  # good for imbalanced datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1) Import (preprocessed) Dataset
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + os.sep + "data" + os.sep + "mbti_preprocessed_complete.csv"
df = pd.read_csv(path, index_col=0)

y_mult_16 = df.iloc[:, 3].values
y_bin = df.iloc[:, 5].values
X = df.iloc[:, 4].values


# print("Variable to be classified/predicited: \n" + str(y) + "\n")
# print("Train Data: \n" + str(X) + "\n")

# Show Column and Feature Names
# print("Column Names: \n" + str(df.columns.tolist()) + "\n")

# 2) Vectorize Feautures

# 2.1) pre-trained embeddings
# to-do

# 2.2) own-trained embeddings

# Word2Vec (in detail see: https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381)

def create_embeddings(X_Data, y_Data_mult, y_Data_bin):
    """
    :param X_Data: text input data (posts)
    :param y_mult: 16 personality types encoded
    :param y_bin: extroverts and introverts encoded
    :return: embeddings for training and testing (X_train_emb_mult, X_train_emb_bin, y_train_emb_mult, y_train_emb_bin,
                                                 (X_test_emb_mult, X_test_emb_bin, y_test_emb_mult, y_test_emb_bin)
    """
    X_train_own_emb_mult, X_test_own_emb_mult, y_train_emb_mult, y_test_emb_mult = train_test_split(X_Data, y_Data_mult,
                                                                                                    test_size=0.3,
                                                                                                    random_state=42069)
    X_train_own_emb_bin, X_test_own_emb_bin, y_train_emb_bin, y_test_emb_bin = train_test_split(X_Data, y_Data_bin,
                                                                                                test_size=0.3,
                                                                                                random_state=42069)

    # For Binary classification
    w2v_model = gensim.models.Word2Vec(X_train_own_emb_bin,
                                       vector_size=100,
                                       window=5,
                                       min_count=2)

    # Generate contextualized vectors for each word
    words = set(w2v_model.wv.index_to_key)

    X_train_vect_own_emb_bin = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                         for ls in X_train_own_emb_bin])

    X_test_vect_own_emb_bin = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                        for ls in X_test_own_emb_bin])

    # Averaging word vectors for same sentence length
    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    X_train_emb_bin = []
    for v in X_train_vect_own_emb_bin:
        if v.size:
            X_train_emb_bin.append(v.mean(axis=0))
        else:
            X_train_emb_bin.append(np.zeros(100, dtype=float))

    X_test_emb_bin = []
    for v in X_test_vect_own_emb_bin:
        if v.size:
            X_test_emb_bin.append(v.mean(axis=0))
        else:
            X_test_emb_bin.append(np.zeros(100, dtype=float))

    # For Multi classification
    w2v_model = gensim.models.Word2Vec(X_train_own_emb_mult,
                                       vector_size=100,
                                       window=5,
                                       min_count=2)

    # Generate contextualized vectors for each word
    words = set(w2v_model.wv.index_to_key)

    X_train_vect_own_emb_mult = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                          for ls in X_train_own_emb_mult])

    X_test_vect_own_emb_mult = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                         for ls in X_test_own_emb_mult])

    # Averaging word vectors for same sentence length
    X_train_emb_mult = []
    for v in X_train_vect_own_emb_mult:
        if v.size:
            X_train_emb_mult.append(v.mean(axis=0))
        else:
            X_train_emb_mult.append(np.zeros(100, dtype=float))

    X_test_emb_mult = []
    for v in X_test_vect_own_emb_mult:
        if v.size:
            X_test_emb_mult.append(v.mean(axis=0))
        else:
            X_test_emb_mult.append(np.zeros(100, dtype=float))

    return [X_train_emb_mult,
            X_train_emb_bin,
            y_train_emb_mult,
            y_train_emb_bin,
            X_test_emb_mult,
            X_test_emb_bin,
            y_test_emb_mult,
            y_test_emb_bin]


# Create own embeddings
# X_train_emb_mult = create_embeddings(X, y_mult_16, y_bin)[0]
# X_train_emb_bin = create_embeddings(X, y_mult_16, y_bin)[1]
# y_train_emb_mult = create_embeddings(X, y_mult_16, y_bin)[0]
# y_train_emb_bin = create_embeddings(X, y_mult_16, y_bin)[0]
# X_test_emb_mult = create_embeddings(X, y_mult_16, y_bin)[0]
# X_test_emb_bin = create_embeddings(X, y_mult_16, y_bin)[0]
# y_test_emb_mult = create_embeddings(X, y_mult_16, y_bin)[0]
# y_test_emb_bin = create_embeddings(X, y_mult_16, y_bin)[0]

# 2.3) TFIDF
tfidfconverter = TfidfVectorizer(use_idf=True)
X = tfidfconverter.fit_transform(X).toarray()

# Compare tokens and vectors
# print("First post tokens: " + df["preprocessed_posts"][0])
# print("Vectorized tokens of first post: " + str(X[0]))

# 3) Create Train and Test Split
X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(X, y_mult_16, test_size=0.3, random_state=42069)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42069)


# Show Train and Test Data
# print("Train Data of X: " + "\n" + str(X_train) + "\n" + "X Train Examples: " + str(len(X_train)) + "\n")
# print("Test Data of X: " + "\n" + str(X_test) + "\n" + "X Test Examples: " + str(len(X_test)) + "\n")
# print("Train Data of y_mult: " + "\n" + str(y_train_mult) + "\n" + "Y Train Examples: " + str(len(y_train_mult)) + "\n")
# print("Test Data of y_mult: " + "\n" + str(y_test_mult) + "\n" + "Y Test Examples: " + str(len(y_test_mult)) + "\n")
# print("Train Data of y_bin: " + "\n" + str(y_train_bin) + "\n" + "Y Train Examples: " + str(len(y_train_bin)) + "\n")
# print("Test Data of y_bin: " + "\n" + str(y_test_bin) + "\n" + "Y Test Examples: " + str(len(y_test_bin)) + "\n")

# 4) Normalization to reduce bias i.e. hence outliners; all are distributed equally (only normalize train data!).
# X_train_scaled = StandardScaler().fit_transform(X_train)


# 5) Build models for classification

# GaussianNB
def classifying(classifier, name, X_train, X_test, y_train, y_test, mult=True, grid=True):

    """
    :param X_train: training data or features (posts)
    :param X_test: test data
    :param y_train: training data for classifying post to personality type (binary and multi)
    :param y_test: test data
    :param mult: if true multi-classification of all 16 personality types will be done
    :param grid: if true hyperparameters of Naive Bayes models will be optimized
    :return: prints model evalutions and best estimators from Gridsearch
    """

    # Classifier
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(name + " Model Evaluation")
    print("---------------------------")
    print()
    print("Confusion Matrix")
    cnf = confusion_matrix(y_test, y_pred)
    print(cnf)
    print()

    if mult:
        print("Multi-Classification Report for 16 personality types")
        print(classification_report(y_test, y_pred, target_names=df['type'].unique()))
        print()
        print("Micro F1-Score")
        print(f1_score(y_test, y_pred, average="micro"))
        print()
        print("Macro F1-Score")
        print(f1_score(y_test, y_pred, average="macro"))
        print()

        if grid:
            # Tune Hyperparameter with Gridsearch
            param_grid = {
                'var_smoothing': np.logspace(0, -9, num=100),
            }
            classifier_1_tuned = GridSearchCV(estimator=GaussianNB(),
                                              param_grid=param_grid,
                                              verbose=1,
                                              cv=10,  # 10-Fold
                                              n_jobs=-1)  # Use all CPU Kernels
            classifier_1_tuned.fit(X_train, y_train)
            print(classifier_1_tuned.best_estimator_)

        else:
            print("No Gridsearch will be done, hence 'grid = False' ")
            print("")

    else:
        print("Binary-Classification Report for extroverts and introverts")
        print(classification_report(y_test, y_pred, target_names=["Introvert", "Extrovert"]))
        print()
        print("Binary F1-Score")
        print(f1_score(y_test, y_pred, average="binary"))
        print()

        if grid:
            # Tune Hyperparameter with Gridsearch
            param_grid = {
                'var_smoothing': np.logspace(0, -9, num=100),
            }
            classifier_1_tuned = GridSearchCV(estimator=GaussianNB(),
                                              param_grid=param_grid,
                                              verbose=1,
                                              cv=10,  # 10-Fold
                                              n_jobs=-1)  # Use all CPU Kernels
            classifier_1_tuned.fit(X_train, y_train)
            print(classifier_1_tuned.best_estimator_)

        else:
            print("No Gridsearch will be done, hence 'grid = False'")
            print("")

    print("End of " + name + " Evaluation")
    print("----------------------------")
    print()

# Evaluating all Naive Bayes Classifiers for Binary- and Multi-Classification with/without Gridsearch
classifiers = {"GaussianNB": GaussianNB(),
               "BernoulliNB": BernoulliNB(),
               "MultinomialNB": MultinomialNB(),
               "ComplementNB": ComplementNB(),
               "CategoricalNB": CategoricalNB()}

try:
    for name, classifier in classifiers.items():

        # Binary Evaluation
        classifying(classifier, name, X_train_bin, X_test_bin, y_train_bin, y_test_bin, mult=False, grid=False)

        # Multi Evaluation
        classifying(classifier, name, X_train_bin, X_test_bin, y_train_bin, y_test_bin, mult=True, grid=False)

except Exception as inst:
    print(inst)
    pass

    """
    IndexError for CategoricalNB():
    "IndexError: index 1 is out of bounds for axis 1 with size 1"
    --> Clarification: https://github.com/scikit-learn/scikit-learn/issues/16028
    """