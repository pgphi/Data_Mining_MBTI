import os
import pandas as pd
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import re
import tldextract
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# import raw dataset
df = pd.read_csv("data/mbti_1.csv")
print(df.head())


# preprocess raw data

# Remove non-alpha characters and preservers whitespaces (me420 i'm done that's it!!! --> me im done thats it)
def split_posts(posts: str) -> str:
    return posts.replace("|||", " ")


# Replaces URLs with Second-Level-Domain (https://www.youtube.com/watch?v=dQw4w9WgXcQ --> youtube)
def replace_url_with_domain(posts: str) -> str:

    url_list = re.findall(r'(https?://[^\s]+)', posts)

    if len(url_list) > 0:
        for url in url_list:
            domain = tldextract.extract(url).domain
            posts = posts.replace(url, domain)

    return posts


# Corrects expressive lengthening / word lengthening (hellooo --> hello)
def correct_expressive_lengthening(posts: str) -> str:
    re.sub(r'(.)\1{3,}', r'\1', posts)
    return posts


# Tokenization: ["Hello my name is"] --> ["Hello", "my", "name", "is"]
def tokenize_posts(posts: str) -> list[str]:
    tokens = nltk.word_tokenize(posts.lower())
    return tokens


# remove stopwords i.e. "and"
def remove_stopwords(tokens):
    stopwords_ = set(stopwords.words("english"))

    # https://escholarship.org/content/qt6n5652cx/qt6n5652cx.pdf
    paper_words = ["got", "a", "i", "il", "be", "the", "of", "do", "not", "can", "am"]

    custom_stopwords = stopwords_ - set(paper_words)

    clean_tokens = []
    for token in tokens:
        if token not in custom_stopwords:
            clean_tokens.append(token)

    return clean_tokens


# Keep context of word i.e. changing --> change (instead of changing --> chang, when using stemming)
def lemmatize_posts(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in tokens:
        # print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))
        lemmas.append(lemmatizer.lemmatize(w))
    return lemmas


# Data exploration
def explore(corpus):
    frequency_distribution = FreqDist(corpus)
    plt.ion()
    frequency_distribution.plot(30, title="Most 30-Words for " + type + " Types", cumulative=False)
    plt.savefig("img/" + type + "_frequency.png")
    plt.ioff()
    plt.show()
    plt.ion()
    word_cloud = WordCloud(collocations=False, background_color='white').generate()
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("img/" + type + "_wordcloud.png")
    plt.ioff()
    plt.show()


def group_prepro_data(df):
    """
    :param df: (preprocessed) data
    :return: create csv file for every MBTI type
    """

    df_grouped = df.groupby(by="type")

    for perso_type, data in df_grouped:

        if perso_type == "ENFJ":
            df_ENFJ = pd.DataFrame(data)
            df_ENFJ.to_csv("data/personality_types/df_ENFJ.csv")
            print(perso_type)
            print(df_ENFJ)
            print(data["posts"])

        elif perso_type == "ENFP":
            df_ENFP = pd.DataFrame(data)
            df_ENFP.to_csv("data/personality_types/df_ENFP.csv")
            print(perso_type)
            print(df_ENFP)

        elif perso_type == "ENTJ":
            df_ENTJ = pd.DataFrame(data)

            df_ENTJ.to_csv("data/personality_types/df_ENTJ.csv")
            print(perso_type)
            print(df_ENTJ)

        elif perso_type == "ENTP":
            df_ENTP = pd.DataFrame(data)
            df_ENTP.to_csv("data/personality_types/df_ENTP.csv")
            print(perso_type)
            print(df_ENTP)

        elif perso_type == "ESFJ":
            df_ESFJ = pd.DataFrame(data)
            df_ESFJ.to_csv("data/personality_types/df_ESFJ.csv")
            print(perso_type)
            print(df_ESFJ)

        elif perso_type == "ESFP":
            df_ESFP = pd.DataFrame(data)
            df_ESFP.to_csv("data/personality_types/df_ESFP.csv")
            print(perso_type)
            print(df_ESFP)

        elif perso_type == "ESTP":
            df_ESTP = pd.DataFrame(data)
            df_ESTP.to_csv("data/personality_types/df_ESTP.csv")
            print(perso_type)
            print(df_ESTP)

        elif perso_type == "INFJ":
            df_INFJ = pd.DataFrame(data)
            df_INFJ.to_csv("data/personality_types/df_INFJ.csv")
            print(perso_type)
            print(df_INFJ)

        elif perso_type == "INFP":
            df_INFP = pd.DataFrame(data)
            df_INFP.to_csv("data/personality_types/df_INFP.csv")
            print(perso_type)
            print(df_INFP)

        elif perso_type == "INTJ":
            df_INTJ = pd.DataFrame(data)
            df_INTJ.to_csv("data/personality_types/df_INTJ.csv")
            print(perso_type)
            print(df_INTJ)

        elif perso_type == "INTP":
            df_INTP = pd.DataFrame(data)
            df_INTP.to_csv("data/personality_types/df_INTP.csv")
            print(perso_type)
            print(df_INTP)

        elif perso_type == "ISFJ":
            df_ISFJ = pd.DataFrame(data)
            df_ISFJ.to_csv("data/personality_types/df_ISFJ.csv")
            print(perso_type)
            print(df_ISFJ)

        elif perso_type == "ISFP":
            df_ISFP = pd.DataFrame(data)
            df_ISFP.to_csv("data/personality_types/df_ISFP.csv")
            print(perso_type)
            print(df_ISFP)

        elif perso_type == "ISTJ":
            df_ISTJ = pd.DataFrame(data)
            df_ISTJ.to_csv("data/personality_types/df_ISTJ.csv")
            print(perso_type)
            print(df_ISTJ)

        elif perso_type == "ISTP":
            df_ISTP = pd.DataFrame(data)
            df_ISTP.to_csv("data/personality_types/df_ISTP.csv")
            print(perso_type)
            print(df_ISTP)


# create dataset with binary labels
def binary_labeling():
    """
    :param:
    :return:
    Extroverted - 1
    Introverted - 0
    """

    # List of all datasets
    list = os.listdir("data/personality_types")

    # create csv datasets
    df_introverted_binary = pd.DataFrame()
    df_extroverted_binary = pd.DataFrame()

    for csv in list:

        if csv == "df_ENFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENFJ", value=1, inplace=True)

        elif csv == "df_ENFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENFP", value=1, inplace=True)

        elif csv == "df_ENTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENTJ", value=1, inplace=True)

        elif csv == "df_ENTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENTP", value=1, inplace=True)

        elif csv == "df_ESFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESFJ", value=1, inplace=True)

        elif csv == "df_ESFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESFP", value=1, inplace=True)

        elif csv == "df_ESTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESTP", value=1, inplace=True)

        elif csv == "df_INFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INFJ", value=0, inplace=True)

        elif csv == "df_INFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INFP", value=0, inplace=True)

        elif csv == "df_INTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INTJ", value=0, inplace=True)

        elif csv == "df_INTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INTP", value=0, inplace=True)

        elif csv == "df_ISFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISFJ", value=0, inplace=True)

        elif csv == "df_ISFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISFP", value=0, inplace=True)

        elif csv == "df_ISTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISTJ", value=0, inplace=True)

        elif csv == "df_ISTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISTP", value=0, inplace=True)

        # create csv files
        df_binary_all = df_introverted_binary.append(df_extroverted_binary)
        df_binary_all.to_csv("data/df_binary_all.csv")
        df_extroverted_binary.to_csv("data/df_extroverted_binary.csv")
        df_introverted_binary.to_csv("data/df_introverted_binary.csv")


# call Functions

# group_prepro_data(df)
# binary_labeling()
df["preprocessed_posts"] = df["posts"].apply(split_posts)
print(df.head())
df["preprocessed_posts"] = df["preprocessed_posts"].apply(replace_url_with_domain)
print(df.head())
df["preprocessed_posts"] = df["preprocessed_posts"].apply(correct_expressive_lengthening)
print(df.head())
df["preprocessed_posts"] = df["preprocessed_posts"].apply(tokenize_posts)
print(df.head())
df["preprocessed_posts"] = df["preprocessed_posts"].apply(remove_stopwords)
print(df.head())
df["preprocessed_posts"] = df["preprocessed_posts"].apply(lemmatize_posts)
print(df.head())
corpus = df["preprocessed_posts"].apply(lemmatize_posts) # create bag-of-words (corpus/vocabulary)
print(corpus)
# create csv dataset file and add the preprocessed column of text (posts) data
df_preprocessed = df.to_csv("data/df_preprocessed.csv")
print(df_preprocessed.head())
