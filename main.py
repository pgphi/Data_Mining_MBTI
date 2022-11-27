# For Data
import pandas as pd

# For Preparation
from preparation import preprocessing, exploration, visualize_3D_context

# For Classification
from classification import train_test_split, feature_generator, \
    BERT_feature_matrix, naiveBayes, BERT_classifier, neural_net_classifier


if __name__ == "__main__":

    # Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
    N = 1000  # len of dataset 8675


    # import raw dataset
    df = pd.read_csv("data/mbti_1.csv")[0:N]


    # create new csv file of dataset with added preprocessed text data
    df["preprocessed_text"] = df["posts"].apply(lambda x: preprocessing(x))
    df.to_csv("data/df_multi_preprocessed.csv")


    # Create Train and Test Split
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(df, 0.3, 42)


    # Data Exploration
    #preprocessed_text = df["preprocessed_text"]
    #exploration(df, preprocessed_text, 50)
    #visualize_3D_context(preprocessed_text, "think", 300, 5, 5)


    # Feature Generation and Vectorizing (TFIDF, BOW or Embeddings) of Training corpus for Classification
    #X_train_vec, X_test_pad_seq, vectorizer = feature_generator(X_train, X_test, y_train, 2, "tfidf")
    X_train_Feature_Matrix = BERT_feature_matrix(X_train)
    X_test_Feature_Matrix = BERT_feature_matrix(X_test)


    # Classification
    #naiveBayes(X_train_vec, y_train, X_test, y_test, vectorizer)  # use bow or tfidf for vectorizer!
    # neural_net_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer)  # use embeddings for vectorizer!
    BERT_classifier(X_train_Feature_Matrix, X_test_Feature_Matrix, y_train, y_test) # use BERT feature matrix
