from datetime import time
import streamlit as st
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn import manifold


def word_context(word):
    w2v_model = pickle.load(open('models/w2v_model.pkl', 'rb'))

    ## Visualize word and its context in 3D Vector Space
    fig = plt.figure()

    ## word embedding
    embedding = w2v_model.wv[word]

    return embedding

"""
def userinterface():
    st.subheader("Analyzing The Word Context based on Personality Types 💬")
    st.write("University of Mannheim - DMI - Prof. Paulheim")
    st.caption("By Stefan, Mariam, Fabian, Ricarda, Priscilla and Philipp")
    st.markdown("***")
    word = st.text_input("Type in your favorite word (lower case - case sensitive!):", placeholder="i.e. thinking")
    st.success("Stay patient. It takes a few seconds for the model to make a prediction.")

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
        st.info("Word not in corpus or upper case (case sensitive!). Try using another word.")
"""

word_context("I dont know what to do")

