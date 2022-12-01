

      Authors: Ricarda Reiner (1557371), Stefan Stingl (1869334), Fabian Rajwa (1687954),
      Philipp Ganster (1554076), Priscilla Chyrva (1666533), Mariam Arustashvili (1939130)

      submitted to the
      Data and Web Science Group
      Prof. Dr. Heiko Paulheim
      University of Mannheim
      December 2022

---


### About this Project

#### Motivation
The purpose of this project is, to classify the MBTI 16 personality types based on a given text input.
   
    i.e. "This evening i will enjoy myself at home drinking a cup of tea" --> "INFP"
    
![User Interface](https://github.com/pgphi/Data_Mining_MBTI/blob/main/img/ui.png)
    
#### Problem
In order to classify the different personality types, we use a multi-classification approach. Furthermore,
we use the following paradigms for solving the multi-classification problem:

##### Models for Classification

1) Naive Bayes Model (Baseline)
2) Decision Trees	Random Forest	
3) Logistic Regression	
4) LVM
5) kNN
6) Nearest Centroids	
7) XGBoost
8) Skip-Gram Model
9) Transformers (BERT)




### Logic of main.py

    # Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
    N = 8675  # len of dataset 8675 Users with each various posts


    # import raw dataset
    df = pd.read_csv("data/mbti_1.csv")[0:N]


    # create new csv file of dataset with added preprocessed text data
    df["preprocessed_text"] = df["posts"].apply(lambda x: preprocessing(x))
    df.to_csv("data/df_multi_preprocessed.csv")


    # Create Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(df, 0.3, 42069, balancing=True, binary=False)

    pd.set_option('display.max_columns', None)

    """
    print(X_train[0:5])
    print(X_test[0:5])
    print(y_train[0:5])
    print(y_test[0:5])
    """

    # Data Exploration
    # preprocessed_text = df["preprocessed_text"]
    # exploration(X_train, preprocessed_text, 100)
    # visualize_3D_context(preprocessed_text, "think", 300, 5, 5)


    # Feature Generation and Vectorizing (TFIDF, BOW or Embeddings) of Training corpus for Classification
    X_train_vec, X_test_pad_seq, vectorizer = feature_generator(X_train, X_test, y_train, 2, "tfidf", 0.95,
                                                                10000, maxsenlen=100)

    # X_train_Feature_Matrix = BERT_Features(X_train, False, max_len=30)
    # X_test_Feature_Matrix = BERT_Features(X_test, False, max_len=30)


    # Classification
    naiveBayes(X_train_vec, y_train, X_test, y_test, vectorizer, binary=False, k_fold=True)  # use bow or tfidf for vectorizer!
    # NN_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer, binary=True, maxlen=50, epoch=1)  # use different embeddings for vectorizer!
    # BERT_classifier(X_train_Feature_Matrix, X_test_Feature_Matrix, y_train, y_test, binary=False, epoch=3, maxlen=30) # use BERT Features!

