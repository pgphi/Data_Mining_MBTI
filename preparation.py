# For Preprocessing
import tldextract
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# For Data Exploration
from nltk.probability import FreqDist
from sklearn import manifold
from wordcloud import WordCloud

## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re

## for bag-of-words
from gensim.models import Word2Vec, Doc2Vec, FastText
from sklearn import feature_extraction, feature_selection, model_selection

## for deep learning
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import models, layers, preprocessing as kprocessing

## for bert language model
import transformers


def preprocessing(posts):
    """
    :param posts: one row of raw text column of dataset. This function will be applied to pandas DataFrame and
    iterated through everyrow with lambda function.
    :return: cleaned text <string> column of dataset for every raw text row.
    """

    # print("raw post: \n" + posts)

    # preprocess raw data
    # Remove non-alpha characters and preservers whitespaces (me420 i'm done that's it!!! --> me im done thats it)
    posts = posts.replace("|||", " ")
    # print("remove non-alpha chars and whitespaces: \n" + posts)

    # Replaces URLs with Second-Level-Domain (https://www.youtube.com/watch?v=dQw4w9WgXcQ --> youtube)
    url_list = re.findall(r'(https?://[^\s]+)', posts)

    if len(url_list) > 0:
        for url in url_list:
            domain = tldextract.extract(url).domain
            posts = posts.replace(url, domain)

    # print("replace urls with domain name: \n" + posts)

    # handle emojis regex match 2 colons : around a string and replace with just the string
    re.sub(r':(.*?):', r'\1' + "emoji", posts)
    # print("turn emojis into string: \n" + posts)

    # Removes non alphabet chars and \s for matching whitespace
    regex = re.compile('[^a-zA-Z\s]')
    posts = regex.sub('', posts)
    # print("remove more non-alpha chars: \n" + posts)

    # Corrects expressive lengthening / word lengthening (hellooo --> hello)
    posts = re.sub(r'(.)\1{3,}', r'\1', posts)
    # print("correct expressive length: \n" + posts)

    # Tokenization: ["Hello my name is"] --> ["Hello", "my", "name", "is"]
    tokens = nltk.word_tokenize(posts.lower())
    # print("create tokens: \n" + str(tokens))

    # remove stopwords i.e. "and"
    stopwords_ = set(stopwords.words("english"))

    # https://escholarship.org/content/qt6n5652cx/qt6n5652cx.pdf
    paper_words = ["got", "a", "i", "il", "be", "the", "of", "do", "not", "can", "am"]

    custom_stopwords = stopwords_ - set(paper_words)

    clean_tokens = []
    for token in tokens:
        if token not in custom_stopwords:
            clean_tokens.append(token)
    # print("stopwords removed tokens: \n" + str(clean_tokens))

    # Keep context of word i.e. changing --> change (instead of changing --> chang, when using stemming)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in tokens:
        # print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))
        lemmas.append(lemmatizer.lemmatize(w))
    # print("lemmatized text corpus: \n" + str(lemmas))

    # concat lemmas back to text
    clean_text = " ".join(lemmas)

    return clean_text


# Data Exploration
def exploration(df, corpus, n):
    """
    :param df: data set
    :param corpus: rows of preprocessed text column from dataset
    :param n: top N-frequent words
    :return: top N-words frequency graph and wordcloud of preprocessed text and plot distribution of dataset
    """

    for i in corpus:
        corpus = "".join(i)

    tokens = nltk.word_tokenize(corpus.lower())
    # print(tokens)

    frequency_distribution = FreqDist(tokens)
    plt.ion()
    frequency_distribution.plot(n, title="Most " + str(n) + "-Words in corpus", cumulative=False)
    plt.savefig("img/" + "word_frequency.png")
    plt.ioff()
    plt.show()
    plt.ion()

    word_cloud = WordCloud(collocations=False, background_color='white').generate(corpus)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("img/" + "corpus_wordcloud.png")
    plt.ioff()
    plt.show()

    # Plot distribution of dataset - (un)balanced?
    fig, ax = plt.subplots()
    fig.suptitle("Dataset Distribution", fontsize=12)
    df.reset_index().groupby("type").count().sort_values(by=
                                                         "index").plot(kind="barh", legend=False,
                                                                       ax=ax).grid(axis='x')
    plt.savefig("img/" + "dataset_distribution.png")
    plt.ioff()
    plt.show()


def visualize_3D_context(corpus, word, vector, window, epochs):
    """
    :param epochs: no. of training iterations
    :param window: no. of words to be considered for context
    :param vector: vector size of word
    :param corpus: rows of preprocessed text data from dataset
    :param word: word from corpus
    :return: 3D vector space visualization of word and it's context words
    """

    ## create list of lists of unigrams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i + 1])
                     for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)
    print("List of unigrams:")
    print(lst_corpus[0:1])

    ## fit w2v
    w2v_model = Word2Vec(lst_corpus, vector_size=vector, window=window, min_count=1, sg=1, epochs=epochs)
    w2v_model.save("models/w2v_word_context_v2")
    # w2v_model = Word2Vec.load("models/w2v_word_context_v2")

    ## Visualize word and its context in 3D Vector Space
    fig = plt.figure()

    ## word embedding
    tot_words = [word] + [tupla[0] for tupla in w2v_model.wv.most_similar(word, topn=20)]
    print(tot_words)
    X = w2v_model.wv[tot_words]

    ## PCA to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=5, n_components=3, init='pca')
    X = pca.fit_transform(X)

    ## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x", "y", "z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1

    ## plot 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"] == 0]['x'],
               dtf_[dtf_["input"] == 0]['y'],
               dtf_[dtf_["input"] == 0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"] == 1]['x'],
               dtf_[dtf_["input"] == 1]['y'],
               dtf_[dtf_["input"] == 1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
           yticklabels=[], zticklabels=[])
    for label, row in dtf_[["x", "y", "z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)
    plt.savefig("img/" + word + "_3D_context.png")
    plt.ioff()
    plt.show()


def train_test_split(df, test_size, rs, balancing, binary):
    """
    :param balancing: if true return balanced train set
    :param binary: if true return 0 (extroverted) and 1 (introverted) classes
    :param df: (preprocessed) dataset
    :param test_size: choose train/test split ratio <float>
    :param rs: choose random state <float>
    :return: Train and Test Split with and without encoding of <string> labels
    """

    enc = OrdinalEncoder()
    df["encoded_types"] = enc.fit_transform(df[["type"]])
    # print("Raw Dataset:")
    # print(df)

    if binary:

        df = df.replace({"type": {"INTJ": "Introverted",
                                "INTP": "Introverted",
                                "ENTJ": "Extroverted",
                                "ENTP": "Extroverted",
                                "INFJ": "Introverted",
                                "INFP": "Introverted",
                                "ENFJ": "Extroverted",
                                "ENFP": "Extroverted",
                                "ISTJ": "Introverted",
                                "ISFJ": "Introverted",
                                "ESTJ": "Extroverted",
                                "ESFJ": "Extroverted",
                                "ISTP": "Introverted",
                                "ISFP": "Introverted",
                                "ESTP": "Extroverted",
                                "ESFP": "Extroverted"}})



    # Create training and test split
    ## get X
    X_train, X_test = model_selection.train_test_split(df, test_size=test_size, random_state=rs)
    # print("X Training Set:" + "länge=" + str(len(X_train)))
    # print(X_train)
    # print("X Test Set:" + "länge=" + str(len(X_test)))
    # print(X_test)

    ## get target
    y_train = X_train["type"].values
    print("Y Train Set:" + "länge=" + str(len(y_train)))
    # print(y_train)
    y_test = X_test["type"].values
    print("Y Test Set:" + "länge=" + str(len(y_test)))
    # print(y_test)

    # Balancing
    if balancing:
        oversample = RandomOverSampler(sampling_strategy="minority")
        X_over, y_over = oversample.fit_resample(X_train, y_train)
        X_train = X_over
        y_train = y_over

    else:
        None

    return X_train, X_test, y_train, y_test


def feature_generator(X_train, X_test, y_train, n_gram, vectorizer, p_value):

    """
    :param X_train: training data
    :param X_test: test data
    :param y_train: target data
    :param n_gram: set <integer> for bigram, trigram etc.
    :param vectorizer: choose vectorizer <string> ("tfidf", "bow", "w2v_embedding" or "fasttext")
    :param p-value: Set threshold for vocabulary size (the higher the lower the size, but more correlated words v.v.)
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
        # print("Training vocabulary size before dimension reduction: " + str(len(dic_vocabulary)))

        ##Create Sparse Matrix: N (No. of Docs. or rows in training set) x 10000 (vocabulary size)
        sns.set()
        sns.heatmap(X_training.todense()[:, np.random.randint(0, X_training.shape[1], 100)] == 0,
                    vmin=0, vmax=1, cbar=False).set_title('TFIDF Sparse Matrix Sample (non-zero values in black)')
        plt.savefig("img/" + "TFIDF_sparse_matrix.png")
        plt.ioff()
        plt.show()

        ##Look up Position of a certain word in the Sparse Matrix
        word = "think"
        # print("Position of the word " + word + " in matrix: " + str(dic_vocabulary[word]))

        # Feature Selection

        ##Reduce Dimensionality for sparse data with Chi-Quadrat
        X_names = TFIDF.get_feature_names_out()
        p_value_limit = p_value
        features = pd.DataFrame()
        # print("Top Features for Each Class:")
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

    elif vectorizer == "w2v_embeddings":

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
                    cbar=False).set_title('W2V Embedding Sparse Matrix Sample (non-zero values in black)')
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

    elif vectorizer == "fasttext":

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

        ## fit fasttext
        fasttext_model = FastText(lst_corpus, vector_size=300, window=5, min_count=1, epochs=5)
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
                    cbar=False).set_title('FastText Embedding Sparse Matrix Sample (non-zero values in black)')
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
                embeddings[idx] = fasttext_model.wv[word]

            ## if word not in model then skip and the row stays all 0s
            except:
                pass

        word = "think"
        print("dic[word]:", dic_vocabulary[word], "|idx")
        print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, "|vector")

        return X_trained_pad_seq, X_test_pad_seq, embeddings


def BERT_Features(X):
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
        idx_new.append(i[:len(i) - 2])
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

    # print(len(segments[0]))
    # print(len(masks[0]))
    # print(len(idx[0]))

    ## Feature Matrix: 3 x 700 (No. of Docs/rows) x 50

    X_train_Feature_Matrix = (np.asarray(idx, dtype='int32'),
                              np.asarray(masks, dtype='int32'),
                              np.asarray(segments, dtype='int32'))

    # i = 0
    # print("txt: ", X_train["preprocessed_text"].iloc[0])
    # print("tokenized:", [tokenizer.convert_ids_to_tokens(idx) for idx in X_train_Feature_Matrix[0][i].tolist()])
    # print("idx: ", X_train_Feature_Matrix[0][i])
    # print("mask: ", X_train_Feature_Matrix[1][i])
    # print("segment: ", X_train_Feature_Matrix[2][i])

    return X_train_Feature_Matrix

# ---------------------------------------------------------------------------------------------------------------------
