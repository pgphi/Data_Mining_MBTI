# For Data
import pandas as pd

# For Preprocessing
import re
import tldextract
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# For Data Exploration
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from sklearn import manifold
from wordcloud import WordCloud



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

# ---------------------------------------------------------------------------------------------------------------------
