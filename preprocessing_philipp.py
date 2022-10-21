import nltk
import pandas as pd
import re

import pylab
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textwrap import wrap


def preprocessing(df, type):
    """
    :param df: csv dataset
    :return: preprocessed raw and clean text and lemmas, as well as frequency plot and wordcloud
    """

    raw_text = "".join(df["posts"])

    # remove non-alphabetic characters and extra spaces
    clean_text = " ".join([w for w in raw_text.split() if w.isalpha()])

    # remove repeated characters i.e. hellooooooo
    clean_text = re.sub(r'(.)\1{3,}', r'\1', clean_text)

    # remove hyperlinks
    clean_text = re.sub(r"https?://\S+", "", clean_text)

    # lowercasting
    clean_text.lower()

    # tokenization and removing stopwords under condition: https://escholarship.org/content/qt6n5652cx/qt6n5652cx.pdf
    tokens = nltk.word_tokenize(clean_text)
    stopwords_ = set(stopwords.words("english"))
    clean_tokens = [t for t in tokens if not t in stopwords_ and not "got" or "a" or "i" or "ll" or "be" or "the"
                    or "of" or "do" or "not" or "can" or "am"]
    clean_text = " ".join(clean_tokens)

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in clean_tokens:
        # print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))
        lemmas.append(lemmatizer.lemmatize(w))
    # print(lemmas)

    # Analysis
    frequency_distribution = FreqDist(lemmas)
    plt.ion()
    frequency_distribution.plot(30, title="Most 30-Words for " + type + " Types", cumulative=False)
    plt.savefig("img/" + type + "_frequency.png")
    plt.ioff()
    plt.show()

    plt.ion()
    word_cloud = WordCloud(collocations=False, background_color='white').generate(clean_text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("img/" + type + "_wordcloud.png")
    plt.ioff()
    plt.show()

    return [raw_text, clean_text, lemmas]


def group_prepro_data(df):
    """
    :param df: (preprocessed) data
    :return: preprocessed csv file for each personality type and evaluations for each type i.e. ENFJ.
    """

    df_grouped = df.groupby(by="type")

    for perso_type, data in df_grouped:

        if perso_type == "ENFJ":
            df_ENFJ = pd.DataFrame(data)
            df_ENFJ.to_csv("data/df_ENFJ.csv")
            print(perso_type)
            print(df_ENFJ)
            print(data["posts"])
            preprocessing(df_ENFJ, "ENFJ")

        elif perso_type == "ENFP":
            df_ENFP = pd.DataFrame(data)
            df_ENFP.to_csv("data/df_ENFP.csv")
            print(perso_type)
            print(df_ENFP)
            preprocessing(df_ENFP, "ENFP")

        elif perso_type == "ENTJ":
            df_ENTJ = pd.DataFrame(data)
            df_ENTJ.to_csv("data/df_ENTJ.csv")
            print(perso_type)
            print(df_ENTJ)
            preprocessing(df_ENTJ, "ENTJ")

        elif perso_type == "ENTP":
            df_ENTP = pd.DataFrame(data)
            df_ENTP.to_csv("data/df_ENTP.csv")
            print(perso_type)
            print(df_ENTP)
            preprocessing(df_ENTP, "ENTP")

        elif perso_type == "ESFJ":
            df_ESFJ = pd.DataFrame(data)
            df_ESFJ.to_csv("data/df_ESFJ.csv")
            print(perso_type)
            print(df_ESFJ)
            preprocessing(df_ESFJ, "ESFJ")

        elif perso_type == "ESFP":
            df_ESFP = pd.DataFrame(data)
            df_ESFP.to_csv("data/df_ESFP.csv")
            print(perso_type)
            print(df_ESFP)
            preprocessing(df_ESFP, "ESFP")

        elif perso_type == "ESTP":
            df_ESTP = pd.DataFrame(data)
            df_ESTP.to_csv("data/df_ESTP.csv")
            print(perso_type)
            print(df_ESTP)
            preprocessing(df_ESTP, "ESTP")

        elif perso_type == "INFJ":
            df_INFJ = pd.DataFrame(data)
            df_INFJ.to_csv("data/df_INFJ.csv")
            print(perso_type)
            print(df_INFJ)
            preprocessing(df_INFJ, "INFJ")

        elif perso_type == "INFP":
            df_INFP = pd.DataFrame(data)
            df_INFP.to_csv("data/df_INFP.csv")
            print(perso_type)
            print(df_INFP)
            preprocessing(df_INFP, "INFP")

        elif perso_type == "INTJ":
            df_INTJ = pd.DataFrame(data)
            df_INTJ.to_csv("data/df_INTJ.csv")
            print(perso_type)
            print(df_INTJ)
            preprocessing(df_INTJ, "INTJ")

        elif perso_type == "INTP":
            df_INTP = pd.DataFrame(data)
            df_INTP.to_csv("data/df_INTP.csv")
            print(perso_type)
            print(df_INTP)
            preprocessing(df_INTP, "INTP")

        elif perso_type == "ISFJ":
            df_ISFJ = pd.DataFrame(data)
            df_ISFJ.to_csv("data/df_ISFJ.csv")
            print(perso_type)
            print(df_ISFJ)
            preprocessing(df_ISFJ, "ISFJ")

        elif perso_type == "ISFP":
            df_ISFP = pd.DataFrame(data)
            df_ISFP.to_csv("data/df_ISFP.csv")
            print(perso_type)
            print(df_ISFP)
            preprocessing(df_ISFP, "ISFP")

        elif perso_type == "ISTJ":
            df_ISTJ = pd.DataFrame(data)
            df_ISTJ.to_csv("data/df_ISTJ.csv")
            print(perso_type)
            print(df_ISTJ)
            preprocessing(df_ISTJ, "ISTJ")

        elif perso_type == "ISTP":
            df_ISTP = pd.DataFrame(data)
            df_ISTP.to_csv("data/df_ISTP.csv")
            print(perso_type)
            print(df_ISTP)
            preprocessing(df_ISTP, "ISTP")


# load data
DF = pd.read_csv("data/mbti_1.csv")

# grouping and preprocessing
group_prepro_data(DF)
