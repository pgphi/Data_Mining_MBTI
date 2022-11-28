## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for bag-of-words
from sklearn import naive_bayes, pipeline

## for deep learning
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras import models, layers, preprocessing as kprocessing

## for bert language model
import transformers


##Classification
def naiveBayes(X_training, y_training, X_testing, y_testing, vectorizer):
    classifier = naive_bayes.MultinomialNB()

    ## pipeline

    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])

    """
    print("Y Training Data:")
    print(y_training)

    print("X Training Data:")
    print(X_training)

    print("Y Test Data:")
    print(y_testing)

    print("X Test Data:")
    print(X_testing)
    """

    ## train classifier
    model["classifier"].fit(X_training, y_training)

    ## test
    X = X_testing["preprocessed_text"].values.astype(str)
    predicted = model.predict(X)
    predicted_prob = model.predict_proba(X)

    ## Evaluation
    classes = np.unique(y_testing)

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_testing, predicted)
    auc = roc_auc_score(y_testing, predicted_prob, multi_class="ovr")
    micro_f1 = f1_score(y_testing, predicted, average="micro")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Micro F1:", round(micro_f1, 2))
    print()
    print("Detail:")
    print(classification_report(y_testing, predicted))

    ## Plot confusion matrix
    cm = confusion_matrix(y_testing, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Naive Bayes Confusion matrix")
    plt.yticks(rotation=0)
    plt.savefig("img/" "naive_bayes_confusion_matrix.png")
    plt.ioff()
    plt.show()


def NN_classifier(X_train_vec, y_train, X_test_pad_seq, y_test, vectorizer, binary=bool):

    """
    :param X_train_vec: Vectorized Training Data
    :param y_train: Target training data
    :param X_test_pad_seq: Padded and sequenced test data
    :param y_test: target test data
    :param vectorizer: embeddings
    :param binary: adjust model and scoring in regard to binary or multi
    :return: evaluation and classification report of neural network
    """

    """
    About the Architecture of this neural network:
    1 - an Embedding layer that takes the sequences as input and the word vectors as weights.
    2 - Two layers of Bidirectional LSTM to model the order of words in a sequence in both directions.
    3 - Two final dense layers that will predict the probability of each MBTI Type.
    """

    ## input
    x_in = layers.Input(shape=(30,))

    ## embedding
    x = layers.Embedding(input_dim=vectorizer.shape[0],
                         output_dim=vectorizer.shape[1],
                         weights=[vectorizer],
                         input_length=30, trainable=False)(x_in)

    ## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=30, dropout=0.2,
                                         return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=30, dropout=0.2))(x)

    ## final dense layers
    x = layers.Dense(64, activation='relu')(x)

    if binary:
        y_out = layers.Dense(2, activation='softmax')(x)

    elif not binary:
        y_out = layers.Dense(16, activation='softmax')(x)

    ## compile
    model = models.Model(x_in, y_out)
    model.save("models/NN_classifier_v3")

    # model = models.Model.load("models/NN_classifier_v3")
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    ## Encode y data ENFP --> 0, ... INFP --> 15
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])
    print(y_train)
    X_train_len = len(X_train_vec)
    y_train_len = len(y_train)

    if X_train_len == y_train_len:
        print("Gleiche Anzahl (Zeilen) der Datensätze gegeben!")
        print("Länge des Trainingsatzes X = " + str(len(X_train_vec)))
        print("Länge des Trainingsatzes Y (Labels) = " + str(len(y_train)))

    elif X_train_len is not y_train_len:
        print("Bitte auf gleiche Anzahl (Zeilen) der Datensätze achten!")
        print("Länge des Trainingsatzes X = " + str(len(X_train_vec)))
        print("Länge des Trainingsatzes Y (Labels) = " + str(len(y_train)))

    ## train deep neural network
    train_neuro_net = model.fit(x=X_train_vec, y=y_train, batch_size=256,
                                epochs=10, shuffle=True, verbose=0,
                                validation_split=0.3)

    ## plot loss and accuracy
    metrics = [k for k in train_neuro_net.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(train_neuro_net.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(train_neuro_net.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(train_neuro_net.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(train_neuro_net.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt._warn_if_gui_out_of_main_thread()
    plt.savefig("img/" "neural_net_loss_accuracy_curves.png")
    plt.ioff()
    plt.show()

    ## test
    predicted_prob = model.predict(X_test_pad_seq)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]

    ## Evaluation
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = accuracy_score(y_test, predicted)

    if binary:
        #print(y_test)
        #print(predicted)
        #print(predicted_prob)

        auc = roc_auc_score(y_test, predicted_prob[: ,1], average="micro")

        micro_f1 = f1_score(y_test, predicted, average="micro")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Micro F1:", round(micro_f1, 2))
        print()
        print("Detail:")
        print(classification_report(y_test, predicted))

        ## Plot confusion matrix
        cm = confusion_matrix(y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
               yticklabels=classes, title="Binary NN Confusion matrix")
        plt.yticks(rotation=0)
        plt.savefig("img/binary_neural_net_confusion_matrix.png")
        plt.ioff()
        plt.show()

    elif not binary:
        #print(y_test)
        #print(predicted)
        #print(predicted_prob)

        auc = roc_auc_score(y_test, predicted_prob, average="macro", multi_class="ovr")

        micro_f1 = f1_score(y_test, predicted, average="micro")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Micro F1:", round(micro_f1, 2))
        print()
        print("Detail:")
        print(classification_report(y_test, predicted))

        ## Plot confusion matrix
        cm = confusion_matrix(y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
               yticklabels=classes, title="Multi NN Confusion matrix")
        plt.yticks(rotation=0)
        plt.savefig("img/multi_neural_net_confusion_matrix.png")
        plt.ioff()
        plt.show()

def BERT_classifier(X_train, X_test, y_train, y_test, binary, epoch):

    """
    :param X_train: Training Data
    :param X_test:  Test Data
    :param y_train: Training Data Target
    :param y_test: Training Data Target
    :param binary: chose whether output is binary (True) or multi-classification (False)
    :param epoch: choose training parameter
    :return: BERT evaluation report
    """

    ## inputs
    idx = layers.Input((50), dtype="int32", name="input_idx")
    masks = layers.Input((50), dtype="int32", name="input_masks")
    segments = layers.Input((50), dtype="int32", name="input_segments")

    ## pre-trained bert
    nlp = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    bert_out, _ = nlp(idx, attention_mask=masks, token_type_ids=segments, return_dict=False)

    ## fine-tuning
    x = layers.GlobalAveragePooling1D()(bert_out)
    x = layers.Dense(64, activation="gelu")(x)
    y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

    ## compile
    model = models.Model((idx, masks, segments), y_out)
    model.save("models/text_multi_classifier_v1")  # load after trained for efficiency

    # load model
    #model = models.load_model("/models/text__multi_classifier_v1")

    for layer in model.layers[:4]:
        layer.trainable = False
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()



    ## encode y
    dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}

    inverse_dic = {v: k for k, v in dic_y_mapping.items()}

    y_train = np.array([inverse_dic[y] for y in y_train])

    training = model.fit(x=(X_train), y=y_train, batch_size=32,
                         epochs=epoch, shuffle=True, verbose=1,
                         validation_split=0.3)

    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]

    ## Evaluation
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall

    if binary:
        accuracy = accuracy_score(y_test, predicted)
        auc = roc_auc_score(y_test, predicted_prob[: ,1], average="micro")
        micro_f1 = f1_score(y_test, predicted, average="micro")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Micro F1:", round(micro_f1, 2))
        print()
        print("Detail:")
        print(classification_report(y_test, predicted))

        ## Plot confusion matrix
        cm = confusion_matrix(y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
               yticklabels=classes, title="Binary BERT Confusion matrix")
        plt.yticks(rotation=0)
        plt.savefig("img/Binary_BERT_confusion_matrix.png")
        plt.ioff()
        plt.show()

    elif not binary:
        accuracy = accuracy_score(y_test, predicted)
        auc = roc_auc_score(y_test, predicted_prob[:], average="macro", multi_class="ovr")
        micro_f1 = f1_score(y_test, predicted, average="micro")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Micro F1:", round(micro_f1, 2))
        print()
        print("Detail:")
        print(classification_report(y_test, predicted))

        ## Plot confusion matrix
        cm = confusion_matrix(y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
               yticklabels=classes, title="Multi BERT Confusion matrix")
        plt.yticks(rotation=0)
        plt.savefig("img/Multi_BERT_confusion_matrix.png")
        plt.ioff()
        plt.show()


# ---------------------------------------------------------------------------------------------------------------------