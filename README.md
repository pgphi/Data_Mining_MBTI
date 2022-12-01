

      Authors: Ricarda Reiner (1557371), Stefan Stingl (1869334), Fabian Rajwa (1687954),
      Philipp Ganster (1554076), Priscilla Chyrva (1666533), Mariam Arustashvili (1939130)

      submitted to the
      Data and Web Science Group
      Prof. Dr. Heiko Paulheim
      University of Mannheim
      December 2022

---


### About this Project

![User Interface](https://github.com/pgphi/Data_Mining_MBTI/blob/main/img/ui.png)

#### Motivation
The purpose of this project is, to classify the MBTI 16 personality types based on a given text input.
   
    i.e. "This evening i will enjoy myself at home drinking a cup of tea" --> "INFP" (Rather introverted, than extroverted)
    
#### Problem
In order to classify the different personality types, we use a multi-classification approach. Furthermore,
we use the following paradigms for solving the multi-classification problem:

##### Models for Classification

1) Naive Bayes Model (**Baseline**)
2) Decision Trees	Random Forest	
3) Logistic Regression	
4) LVM
5) kNN
6) Nearest Centroids	
7) XGBoost
8) Skip-Gram Model
9) Transformers (BERT)




### Logic of main.py In regard to baseline Model and Neural Networks

###### Choose no of rows you want to train model and make classifications
    # Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
    N = 8675  # len of dataset 8675 Users with each various posts

###### Import raw dataset
    # import raw dataset
    df = pd.read_csv("data/mbti_1.csv")[0:N]

###### Preprocess posts from users
    # create new csv file of dataset with added preprocessed text data
    df["preprocessed_text"] = df["posts"].apply(lambda x: preprocessing(x))
    df.to_csv("data/df_multi_preprocessed.csv")

###### Create train and test split and choose in function whether you want to balance (Oversampler with not majority strategy) and whether you want to encode the target variable for binary- (1 - Introverted | 0 - Extroverted) or multi-classification
    # Create Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(df, 0.3, 42069, balancing=True, binary=False)

###### Exploration function outputs the most frequent 100 words. visualize_3D_context function outputs the 20 context words of a given word.
    # Data Exploration
    preprocessed_text = df["preprocessed_text"]
    exploration(X_train, preprocessed_text, 100)
    visualize_3D_context(preprocessed_text, "think", 300, 5, 5, 20)

###### Create and select Features with feature_generator for Naive Bayes and Neural Network and BERT_Features function for BERT Deep Neural Network.
    # Feature Generation and Vectorizing (TFIDF, BOW or Embeddings) of Training corpus for Classification
    X_train_vec, X_test_pad_seq, vectorizer = feature_generator(X_train, X_test, y_train, 2, "tfidf", 0.95,
                                                                10000, maxsenlen=100)
    X_train_Feature_Matrix = BERT_Features(X_train, False, max_len=30)
    X_test_Feature_Matrix = BERT_Features(X_test, False, max_len=30)


 ##### Feature Selection in regard to Multi-Classification:
    
      # 0.0:
        . selected features: 79
        . top features: an enfj,enfj,enfj and,enfjs,enfjs are,the enfj,sorta,welcome welcome,giggle,hug

      # 1.0:
        . selected features: 27
        . top features: an enfp,enfp,enfps,enfp and,enfps are,the enfp,xd,enfp but,sosx,love

      # 2.0:
        . selected features: 90
        . top features: an entj,entj,entjs,entjs are,the entj,entj and,useless,efficient,goal,ego

      # 3.0:
        . selected features: 18
        . top features: an entp,entp,entps,the entp,entps are,entp and,ne,nt,smarter,fuck

      # 4.0:
        . selected features: 666
        . top features: an esfj,and esfjs,briggs mbti,esfj,esfj and,esfj but,esfj guy,esfj is,esfjs,esfjs and

      # 5.0:
        . selected features: 574
        . top features: an esfp,digger,digger blue,esfp,esfp and,esfps,gti,gti using,my gti,sims

      # 6.0:
        . selected features: 763
        . top features: abt,an estj,ap,eagle,estj,estjs,estjs are,hav,never met,pony

      # 7.0:
        . selected features: 292
        . top features: an estp,estp,estp and,estps,interaction have,interview you,the estp,type via,usually turn,via this

      # 8.0:
        . selected features: 15
        . top features: an infj,infj,infjs,the infj,infjs are,infj and,infjs and,infj is,infj but,esfj

      # 9.0:
        . selected features: 29
        . top features: an infp,infp,infps,infps are,the infp,infp and,infp but,esfj,dream,estp

      # 10.0:
        . selected features: 21
        . top features: an intj,intj,intjs,the intj,intjs are,intj is,intj and,intj but,scientific,science

      # 11.0:
        . selected features: 26
        . top features: intp,intps,an intp,the intp,intps are,universe,intp and,the universe,physic,intp but

      # 12.0:
        . selected features: 124
        . top features: an isfj,isfj,isfjs,the isfj,isfjs are,hello and,teddy,and welcome,isfj and,my istp

      # 13.0:
        . selected features: 61
        . top features: an isfp,isfp,isfps,the isfp,isfp and,isfp but,drawing,isfps are,youtube youtube,handwriting

      # 14.0:
        . selected features: 60
        . top features: an istj,istj,istjs,rant,the istj,istj and,istj but,tax,my girlfriend,deviantart

      # 15.0:
        . selected features: 65
        . top features: an istp,istp,istps,the istp,istps are,istp and,mechanic,fuck,sport,bike

###### Call Classifiers. Every function is given vectorizer (i.e. Embeddings; TFIDF only for naiveBayes). Also choose training parameters (epoch, maxlen) and whether to do k_fold and binary- or multi-classification.
    # Classification
    naiveBayes(X_train_vec, y_train, X_test, y_test, vectorizer, binary=False, k_fold=True)  # use bow or tfidf for vectorizer!
    NN_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer, binary=True, maxlen=50, epoch=1)  # use different embeddings for vectorizer!
    BERT_classifier(X_train_Feature_Matrix, X_test_Feature_Matrix, y_train, y_test, binary=False, epoch=3, maxlen=30) # use BERT Features!
    
