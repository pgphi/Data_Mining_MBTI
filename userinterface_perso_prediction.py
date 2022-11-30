from datetime import time
import streamlit as st
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn import manifold
from preparation import BERT_Features

model = pickle.load(open('models/BERT_model.pkl', 'rb'))

st.subheader("Extrovert - Introvert - Predictor 💬")
st.write("University of Mannheim - DMI - Prof. Paulheim")
st.caption("By Stefan, Mariam, Fabian, Ricarda, Priscilla and Philipp")
st.markdown("***")
personality = st.text_input("How does your perfect friday evening look like?", placeholder="This is a description")
sequence = BERT_Features([personality], production = True, max_len=510)
st.success("Stay patient. It takes a few seconds for the model to make a prediction.")
st.markdown("***")

while personality == False:
    time.sleep(999999)


if personality != "":
    st.info("Based on your answer our fine-tuned BERT Model with the MBTI Kaggle Dataset predicts the following:")
    score = model.predict([sequence])[0][1]


    if score > 0.5:
        st.success("The probability of you being introverted is " + str(score))

    else:
        st.success("The probability of you being extroverted is " + str(1 - score))

else:
    pass