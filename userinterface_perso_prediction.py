from datetime import time
import streamlit as st
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn import manifold


model = pickle.load(open('models/BERT_model.pkl', 'rb'))

st.subheader("Extrovert - Introvert - Predictor 💬")
st.write("University of Mannheim - DMI - Prof. Paulheim")
st.caption("By Stefan, Mariam, Fabian, Ricarda, Priscilla and Philipp")
st.markdown("***")
perso = st.text_input("How does your perfect friday evening look like?", placeholder="This is a description")
prediction = model.predict(["[CLS] " + perso + " [SEP]"])
st.success("Stay patient. It takes a few seconds for the model to make a prediction.")
st.markdown("***")

while perso == False:
    time.sleep(999999)

try:

    if perso != "":
        st.info("Our fine-tuned BERT model with the MBTI Kaggle Dataset predicts that you are an:")
        st.success(prediction)

    else:
        pass

except KeyError:
    st.info("Word not in corpus. Try using another word.")
