from datetime import time
import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn import manifold


def word_context(word):
    
    # for production use pickle hence deployment reasons!!!
    w2v_model = Word2Vec.load("w2v_word_context_v2")

    ## Visualize word and its context in 3D Vector Space
    fig = plt.figure()

    ## word embedding
    tot_words = [word] + [tupla[0] for tupla in w2v_model.wv.most_similar(word, topn=10)]
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
    plt.savefig("img/" + word + "_3D_context.png", dpi=300)


def userinterface():
    
    """
    st.title("Classifying Personality Types 🎭")
    st.subheader("University of Mannheim - DM I - Prof. Paulheim")
    st.write("A Project from Stefan, Mariam, Priscilla, Ricarda, Fabian and Philipp")
    st.markdown("***")
    st.markdown("")
    st.markdown("#### Motivation and Goal")
    st.write(
        "[Yarkoni (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0092656610000541) reported the results "
        "of a large-scale analysis of personality and word use in a large sample of blogs (N = 694). Furthermore, he says that "
        "previous studies have found systematic associations between personality and individual differences in word use. "
        "In Addition [Tskhay and Rule (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0092656613001499)  "
        "examined studies that reported accuracy and consensus effects for the perception of the Big Five traits from "
        "text-based media and online social network websites.")
    st.write(
        "We want to contribute to those findings by using a model which classifies the text input of users and shows "
        "his or her personality type based on [MBTI](https://simple.wikipedia.org/wiki/Myers-Briggs_Type_Indicator)"
        ", as well as the context of a specific word in order to find out the individual differences in word use in regard to personality types.")
    st.info(
        "We used the [(MBTI) Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) "
        "from Kaggle to train our model. For more information on our model see our [Github Repository](https://github.com/pgphi/Data_Mining_MBTI)")
    st.markdown("")
    st.markdown("***")
    st.markdown("")
    """
    
    st.subheader("Analyzing The Word Context based on Personality Types 💬")
    st.write("University of Mannheim - DM I - Prof. Paulheim")
    st.caption("A Project from Stefan, Mariam, Priscilla, Ricarda, Fabian and Philipp")


    word = st.text_input("Type in your favorite word:", placeholder="i.e. thinking", help="only lower case!")

    while word == False:
        time.sleep(999999)


    return word



try:
    word = userinterface()

    if word != "":
        word_context(word)
        st.image("img/" + word + "_3D_context.png")

    else:
        pass

except KeyError:
        st.info("Word not in corpus or lower case. Try using another word or upper case (case sensitive!).")
