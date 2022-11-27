## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse

## for processing
import re

## for bag-of-words
from gensim.models import Word2Vec
from sklearn import feature_extraction, feature_selection, model_selection, naive_bayes, pipeline

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for deep learning
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for bert language model
import transformers


def train_test_split(df, test_size, rs):
    """
    :param df: (preprocessed) dataset
    :param test_size: choose train/test split ratio <float>
    :param rs: choose random state <float>
    :return: Train and Test Split with and without encoding of <string> labels
    """

    enc = OrdinalEncoder()
    df["encoded_types"] = enc.fit_transform(df[["type"]])
    # print("Raw Dataset:")
    # print(df)


    # Create training and test split
    ## get X
    X_train, X_test = model_selection.train_test_split(df, test_size=test_size, random_state=rs)
    # print("X Training Set:" + "länge=" + str(len(X_train)))
    # print(X_train)
    # print("X Test Set:" + "länge=" + str(len(X_test)))
    # print(X_test)


    ## get target
    y_train = X_train["type"].values
    # print("Y Train Set:" + "länge=" + str(len(y_train)))
    # print(y_train)
    y_test = X_test["type"].values
    # print("Y Test Set:" + "länge=" + str(len(y_test)))
    # print(y_test)

    ## encode target ordinally ENJP --> 0 ... INFP --> 15
    y_train_enc = X_train["encoded_types"].values
    # print("Y Encoded Train Set:" + "länge=" + str(len(y_train)))
    # print(y_train)
    y_test_enc = X_test["encoded_types"].values
    # print("Y Encoded Test Set:" + "länge=" + str(len(y_test)))
    # print(y_test)

    return X_train, X_test, y_train, y_test, y_train_enc, y_test_enc


def feature_generator(X_train, X_test, y_train, n_gram, vectorizer):

    """
    :param X_train: training data
    :param X_test: test data
    :param y_train: target data
    :param n_gram: set <integer> for bigram, trigram etc.
    :param vectorizer: choose vectorizer <string> (tfidf, bow or embeddings)
    :return: sparse matrix, re-fitted sparse matrix (chi2), tf/tfidf features and vectorized training data
    """

    # Feature Engineering (vocabulary size = 10.000 words)

    if vectorizer == "tfidf":

        ## TF-IDF (advanced variant of BoW)
        TFIDF = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, n_gram))

        ##Extract Vocabulary
        corpus = X_train["preprocessed_text"].values.astype(str)
        TFIDF.fit(corpus)
        X_training = TFIDF.transform(corpus)
        dic_vocabulary = TFIDF.vocabulary_
        print("Training vocabulary size before dimension reduction: " + str(len(dic_vocabulary)))

        ##Create Sparse Matrix: N (No. of Docs. or rows in training set) x 10000 (vocabulary size)
        sns.set()
        sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                    vmin=0, vmax=1, cbar=False).set_title('TFIDF Sparse Matrix Sample (non-zero values in black)')
        plt.savefig("img/" + "TFIDF_sparse_matrix.png")
        plt.ioff()
        plt.show()

        ##Look up Position of a certain word in the Sparse Matrix
        word = "think"
        print("Position of the word " + word + " in matrix: " + str(dic_vocabulary[word]))

        # Feature Selection

        ##Reduce Dimensionality for sparse data with Chi-Quadrat
        X_names = TFIDF.get_feature_names_out()
        p_value_limit = 0.95
        features = pd.DataFrame()
        print("Top Features for Each Class:")
        for cat in np.unique(y_train):
            chi2, p = feature_selection.chi2(X_training, y_train == cat)
            features = features.append(pd.DataFrame(
                {"feature": X_names, "score": 1 - p, "y": cat}))
            features = features.sort_values(["y", "score"], ascending=[True, False])
            features = features[features["score"] > p_value_limit]
        X_names = features["feature"].unique().tolist()

        for cat in np.unique(y_train):
            print("# {}:".format(cat))
            print("  . selected features:",
                  len(features[features["y"] == cat]))
            print("  . top features:", ",".join(
                features[features["y"] == cat]["feature"].values[:10]))
            print(" ")

        ##Re-Fit vectorizer on corpus with new set of words and create new sparse matrix
        TFIDF = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
        TFIDF.fit(corpus)
        X_train_vec = TFIDF.transform(corpus)
        dic_vocabulary = TFIDF.vocabulary_
        print("Training vocabulary size after dimension reduction: " + str(len(dic_vocabulary)))

        # New Sparse Matrix
        sns.set()
        sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                    vmin=0, vmax=1, cbar=False).set_title(
            'Re-Fit TFIDF Sparse Matrix Sample (non-zero values in black)')
        plt.savefig("img/" + "re_fit_TFIDF_sparse_matrix.png")
        plt.ioff()
        plt.show()

        return X_train_vec, None, TFIDF

    elif vectorizer == "bow":
        ## Count (classic BoW)
        BOW = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1, n_gram))

        ##Extract Vocabulary
        corpus = X_train["preprocessed_text"].values.astype(str)
        BOW.fit(corpus)
        X_training = BOW.transform(corpus)
        dic_vocabulary = BOW.vocabulary_
        print("Training vocabulary size before dimension reduction: " + str(len(dic_vocabulary)))

        ##Create Sparse Matrix: N (No. of Docs. or rows in training set) x 10000 (vocabulary size)
        sns.set()
        sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                    vmin=0, vmax=1, cbar=False).set_title('BOW Sparse Matrix Sample (non-zero values in black)')
        plt.savefig("img/" + "BOW_sparse_matrix.png")
        plt.ioff()
        plt.show()

        ##Look up Position of a certain word in the Sparse Matrix
        word = "think"
        print("Position of the word " + word + " in matrix: " + str(dic_vocabulary[word]))

        # Feature Selection

        ##Reduce Dimensionality for sparse data with Chi-Quadrat
        X_names = BOW.get_feature_names_out()
        p_value_limit = 0.95
        features = pd.DataFrame()
        print("Top Features for Each Class:")
        for cat in np.unique(y_train):
            chi2, p = feature_selection.chi2(X_training, y_train == cat)
            features = features.append(pd.DataFrame(
                {"feature": X_names, "score": 1 - p, "y": cat}))
            features = features.sort_values(["y", "score"], ascending=[True, False])
            features = features[features["score"] > p_value_limit]
        X_names = features["feature"].unique().tolist()

        for cat in np.unique(y_train):
            print("# {}:".format(cat))
            print("  . selected features:",
                  len(features[features["y"] == cat]))
            print("  . top features:", ",".join(
                features[features["y"] == cat]["feature"].values[:10]))
            print(" ")

        ##Re-Fit vectorizer on corpus with new set of words and create new sparse matrix
        BOW = feature_extraction.text.CountVectorizer(vocabulary=X_names)
        BOW.fit(corpus)
        X_train_vec = BOW.transform(corpus)
        dic_vocabulary = BOW.vocabulary_
        print("Training vocabulary size after dimension reduction: " + str(len(dic_vocabulary)))

        # New Sparse Matrix
        sns.set()
        sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                    vmin=0, vmax=1, cbar=False).set_title('Re-fit BOW Sparse Matrix Sample (non-zero values in black)')
        plt.savefig("img/" + "re_fit_BOW_sparse_matrix.png")
        plt.ioff()
        plt.show()

        return X_train_vec, None, BOW

    elif vectorizer == "embeddings":

        corpus = X_train["preprocessed_text"]
        print("Corpus for Word Embeddings:")
        print(corpus[0:1])

        # own-trained with word2vec
        ## create list of lists of unigrams
        lst_corpus = []
        for string in corpus:
            lst_words = string.split()
            lst_grams = [" ".join(lst_words[i:i + 1])
                         for i in range(0, len(lst_words), 1)]
            lst_corpus.append(lst_grams)
        print("List of unigrams:")
        print(lst_corpus[0:1])

        ## detect bigrams and trigrams
        """
        bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
                                                         delimiter=" ".encode(), min_count=5, threshold=10)
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
                                                          delimiter=" ".encode(), min_count=5, threshold=10)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
        """

        ## fit w2v
        w2v_model = Word2Vec(lst_corpus, vector_size=300, window=5, min_count=1, sg=1, epochs=5)
        # visualize_3D_context("enfp")

        # Feature Engineering
        ## tokenize text
        tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                               oov_token="NaN",
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
        dic_vocabulary = tokenizer.word_index

        ## create sequence
        lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

        ## padding sequence
        X_trained_pad_seq = kprocessing.sequence.pad_sequences(lst_text2seq,
                                                               maxlen=30, padding="post", truncating="post")

        # Visualize Sparse Matrix
        sns.set()
        sns.heatmap(X_trained_pad_seq == 0, vmin=0, vmax=1,
                    cbar=False).set_title('Embedding Sparse Matrix Sample (non-zero values in black)')
        plt.ioff()
        plt.show()

        # Feature Engineering for Training Data
        i = 0

        ## list of text: ["I like this", ...]
        len_txt = len(X_train["preprocessed_text"].iloc[i].split())
        print("from: ", X_train["preprocessed_text"].iloc[i], "| len:", len_txt)

        ## sequence of token ids: [[1, 2, 3], ...]
        len_tokens = len(X_trained_pad_seq[i])
        print("to: ", X_trained_pad_seq[i], "| len:", len(X_trained_pad_seq[i]))

        ## vocabulary: {"I":1, "like":2, "this":3, ...}
        print("check: ", X_train["preprocessed_text"].iloc[i].split()[0],
              " -- idx in vocabulary -->", dic_vocabulary[X_train["preprocessed_text"].iloc[i].split()[0]])

        print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

        # Feature Engineering for Test Data
        corpus = X_test["preprocessed_text"]

        ## create list of vectorized n-grams
        lst_corpus = []
        for string in corpus:
            lst_words = string.split()
            lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
            lst_corpus.append(lst_grams)

        # print("Vectorized unigram corpus:")
        print(lst_corpus[0:1])

        ## detect common bigrams and trigrams using the fitted detectors
        # lst_corpus = list(bigrams_detector[lst_corpus])
        # lst_corpus = list(trigrams_detector[lst_corpus])

        ## text to sequence with the fitted tokenizer
        lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
        print("Sequenced unigram corpus:")
        print(lst_text2seq[0:1])

        ## padding sequence
        X_test_pad_seq = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=30, padding="post", truncating="post")

        # Create matrix of embedding for weight matrix in neural network classifier
        ## start the matrix (length of vocabulary x vector size) with all 0s
        embeddings = np.zeros((len(dic_vocabulary) + 1, 300))
        for word, idx in dic_vocabulary.items():
            ## update the row with vector
            try:
                embeddings[idx] = w2v_model.wv[word]

            ## if word not in model then skip and the row stays all 0s
            except:
                pass

        word = "think"
        print("dic[word]:", dic_vocabulary[word], "|idx")
        print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, "|vector")

        return X_trained_pad_seq, X_test_pad_seq, embeddings

def BERT_feature_matrix(X):

    ## bert tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    corpus = X["preprocessed_text"]
    maxlen = 50

    ## add special tokens
    maxqnans = int((maxlen - 20) / 2)
    corpus_tokenized = ["[CLS] " +
                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',
                                                           str(txt).lower().strip()))[:maxqnans]) +
                        " [SEP] " for txt in corpus]

    ## generate masks
    masks = [[1] * len(txt.split(" ")) + [0] * (maxlen - len(
        txt.split(" "))) for txt in corpus_tokenized]

    ## padding
    txt2seq = [txt + " [PAD]" * (maxlen - len(txt.split(" "))) if len(txt.split(" ")) != maxlen else txt for txt in
               corpus_tokenized]

    ## generate idx
    idx = [tokenizer.encode(seq.split(" ")) for seq in txt2seq]

    # For loop for deleting two last elements in list to make the input dimensions of feature matrix the same (50x50x50)
    # Last two elements are most likely "[PAD]", "[PAD]"
    idx_new = []
    for i in idx:
        idx_new.append(i[:len(i)-2])
    idx = idx_new

    ## generate segments
    segments = []
    for seq in txt2seq:
        temp, i = [], 0
        for token in seq.split(" "):
            temp.append(i)
            if token == "[SEP]":
                i += 1
        segments.append(temp)

    #print(len(segments[0]))
    #print(len(masks[0]))
    #print(len(idx[0]))

    ## Feature Matrix: 3 x 700 (No. of Docs/rows) x 50

    X_train_Feature_Matrix = (np.asarray(idx, dtype='int32'),
                              np.asarray(masks, dtype='int32'),
                              np.asarray(segments, dtype='int32'))


    #i = 0
    #print("txt: ", X_train["preprocessed_text"].iloc[0])
    #print("tokenized:", [tokenizer.convert_ids_to_tokens(idx) for idx in X_train_Feature_Matrix[0][i].tolist()])
    #print("idx: ", X_train_Feature_Matrix[0][i])
    #print("mask: ", X_train_Feature_Matrix[1][i])
    #print("segment: ", X_train_Feature_Matrix[2][i])

    return X_train_Feature_Matrix


##Classification
def naiveBayes(X_training, y_training, X_testing, y_testing, vectorizer):
    classifier = naive_bayes.MultinomialNB()

    ## pipeline

    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])

    """
    print("Y Training Data:")
    print(y_training)

    print("X Training Data:")
    print(X_training)

    print("Y Test Data:")
    print(y_testing)

    print("X Test Data:")
    print(X_testing)
    """

    ## train classifier
    model["classifier"].fit(X_training, y_training)

    ## test
    X = X_testing["preprocessed_text"].values.astype(str)
    predicted = model.predict(X)
    predicted_prob = model.predict_proba(X)

    ## Evaluation
    classes = np.unique(y_testing)

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_testing, predicted)
    auc = roc_auc_score(y_testing, predicted_prob, multi_class="ovr")
    micro_f1 = f1_score(y_testing, predicted, average="micro")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Micro F1:", round(micro_f1, 2))
    print()
    print("Detail:")
    print(classification_report(y_testing, predicted))

    ## Plot confusion matrix
    cm = confusion_matrix(y_testing, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Naive Bayes Confusion matrix")
    plt.yticks(rotation=0)
    plt.savefig("img/" "naive_bayes_confusion_matrix.png")
    plt.ioff()
    plt.show()


def neural_net_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer):
    """
    :param X_train_vec: Vectorized Training Data
    :param y_train: Target training data
    :param X_test_pad_seq: Padded and sequenced test data
    :param y_test: target test data
    :param vectorizer: embeddings
    :return: evaluation and classification report of neural network
    """

    """
    About the Architecture of this neural network:
    1 - an Embedding layer that takes the sequences as input and the word vectors as weights.
    2 - Two layers of Bidirectional LSTM to model the order of words in a sequence in both directions.
    3 - Two final dense layers that will predict the probability of each MBTI Type.
    """

    ## input
    x_in = layers.Input(shape=(30,))

    ## embedding
    x = layers.Embedding(input_dim=vectorizer.shape[0],
                         output_dim=vectorizer.shape[1],
                         weights=[vectorizer],
                         input_length=30, trainable=False)(x_in)

    ## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=30, dropout=0.2,
                                         return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=30, dropout=0.2))(x)

    ## final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(16, activation='softmax')(x)

    ## compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    ## Encode y data ENFP --> 0, ... INFP --> 15
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])

    X_train_len = len(X_train_vec)
    y_train_len = len(y_train)

    if X_train_len == y_train_len:
        print("Gleiche Anzahl (Zeilen) der Datensätze gegeben!")
        print("Länge des Trainingsatzes X = " + str(len(X_train_vec)))
        print("Länge des Trainingsatzes Y (Labels) = " + str(len(y_train)))

    elif X_train_len != y_train_len:
        print("Bitte auf gleiche Anzahl (Zeilen) der Datensätze achten!")
        print("Länge des Trainingsatzes X = " + str(len(X_train_vec)))
        print("Länge des Trainingsatzes Y (Labels) = " + str(len(y_train)))

    ## train deep neural network
    train_neuro_net = model.fit(x=X_train_vec, y=y_train, batch_size=256,
                                epochs=10, shuffle=True, verbose=0,
                                validation_split=0.3)

    ## plot loss and accuracy
    metrics = [k for k in train_neuro_net.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(train_neuro_net.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(train_neuro_net.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(train_neuro_net.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(train_neuro_net.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt._warn_if_gui_out_of_main_thread()
    plt.savefig("img/" "neural_net_loss_accuracy_curves.png")
    plt.ioff()
    plt.show()

    ## test
    predicted_prob = model.predict(X_test_pad_seq)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                 predicted_prob]

    ## Evaluation
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_test, predicted)
    auc = roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    micro_f1 = f1_score(y_test, predicted, average="micro")
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
           yticklabels=classes, title="W2V Skip-Gram Confusion matrix")
    plt.yticks(rotation=0)
    plt.savefig("img/neural_net_confusion_matrix.png")
    plt.ioff()
    plt.show()


def BERT_classifier(X_train, X_test, y_train, y_test):
    ## inputs
    idx = layers.Input((50), dtype="int32", name="input_idx")
    masks = layers.Input((50), dtype="int32", name="input_masks")
    segments = layers.Input((50), dtype="int32", name="input_segments")

    ## pre-trained bert
    nlp = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    bert_out, _ = nlp(idx, attention_mask=masks, token_type_ids=segments, return_dict=False)

    ## fine-tuning
    x = layers.GlobalAveragePooling1D()(bert_out)
    x = layers.Dense(64, activation="relu")(x)
    y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

    ## compile
    model = models.Model((idx, masks, segments), y_out)
    for layer in model.layers[:4]:
        layer.trainable = False
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    ## encode y
    dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}

    inverse_dic = {v: k for k, v in dic_y_mapping.items()}

    y_train = np.array([inverse_dic[y] for y in y_train])

    training = model.fit(x=(X_train), y=y_train, batch_size=64,
                         epochs=1, shuffle=True, verbose=1,
                         validation_split=0.3)

    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]

    ## Evaluation
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_test, predicted)
    auc = roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    micro_f1 = f1_score(y_test, predicted, average="micro")
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
           yticklabels=classes, title="BERT Confusion matrix")
    plt.yticks(rotation=0)
    plt.savefig("img/BERT_confusion_matrix.png")
    plt.ioff()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
