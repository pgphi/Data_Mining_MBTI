# For Data
import pandas as pd

# For Preparation
from imblearn.over_sampling import RandomOverSampler

from preparation import preprocessing, exploration, visualize_3D_context, train_test_split, \
    feature_generator, BERT_Features

# For Classification
from NB_SG_BERT import naiveBayes, BERT_classifier, NN_classifier

if __name__ == "__main__":

    # Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
    N = 8675  # len of dataset 8675


    # import raw dataset
    df = pd.read_csv("data/mbti_1.csv")[0:N]


    # create new csv file of dataset with added preprocessed text data
    df["preprocessed_text"] = df["posts"].apply(lambda x: preprocessing(x))
    df.to_csv("data/df_multi_preprocessed.csv")


    # Create Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(df, 0.3, 42069, balancing=False, binary=False)


    # Data Exploration
    # preprocessed_text = df["preprocessed_text"]
    # exploration(X_train, preprocessed_text, 100)
    # visualize_3D_context(preprocessed_text, "think", 300, 5, 5)


    # Feature Generation and Vectorizing (TFIDF, BOW or Embeddings) of Training corpus for Classification
    # X_train_vec, X_test_pad_seq, vectorizer = feature_generator(X_train, X_test, y_train, 2, "w2v_embeddings", 0.95,
                                                                # 10000, maxsenlen=50)

    X_train_Feature_Matrix = BERT_Features(X_train, False, max_len=30)
    X_test_Feature_Matrix = BERT_Features(X_test, False, max_len=30)


    # Classification
    # naiveBayes(X_train_vec, y_train, X_test, y_test, vectorizer, binary=True)  # use bow or tfidf for vectorizer!
    # NN_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer, binary=True, maxlen=50, epoch=1)  # use different embeddings for vectorizer!
    BERT_classifier(X_train_Feature_Matrix, X_test_Feature_Matrix, y_train, y_test, binary=False, epoch=3, maxlen=30) # use BERT Features!
