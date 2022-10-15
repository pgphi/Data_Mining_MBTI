from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from src.utils.utils import check_path_exists, save
import os


class Bert(object):

    def __init__(self):

        self.model = SentenceTransformer(
            "PRE-TRAINED-MODEL") #Search for Modelthat fits

    # Dont Preprocess Texts beforehand! (maybe emojis and stuff out and convert links into )
    def transform(self, raw_texts: pd.Series, store: str = None):


        bert_vec = []

        for text in tqdm(raw_texts):
            embedding = self.model.encode(text)
            bert_vec.append(embedding)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(bert_vec, store)

        return bert_vec