# For Data
import os
import pandas as pd
import numpy as np

# For Preprocessing
import re
import tldextract
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# For Data Exploration
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud



def preprocessing(posts):

    """
    :param posts: one row of raw text column of dataset. This function will be applied to pandas DataFrame and
    iterated through everyrow with lambda function.
    :return: cleaned text <string> column of dataset for every raw text row.
    """

    # print("raw post: \n" + posts)

    # preprocess raw data

    # Remove non-alpha characters and preservers whitespaces (me420 i'm done that's it!!! --> me im done thats it)
    posts = posts.replace("|||", " ")
    # print("remove non-alpha chars and whitespaces: \n" + posts)


    # Replaces URLs with Second-Level-Domain (https://www.youtube.com/watch?v=dQw4w9WgXcQ --> youtube)
    url_list = re.findall(r'(https?://[^\s]+)', posts)

    if len(url_list) > 0:
        for url in url_list:
            domain = tldextract.extract(url).domain
            posts = posts.replace(url, domain)

    # print("replace urls with domain name: \n" + posts)


    # handle emojis regex match 2 colons : around a string and replace with just the string
    re.sub(r':(.*?):', r'\1' + "emoji", posts)
    # print("turn emojis into string: \n" + posts)


    # Removes non alphabet chars and \s for matching whitespace
    regex = re.compile('[^a-zA-Z\s]')
    posts = regex.sub('', posts)
    # print("remove more non-alpha chars: \n" + posts)


    # Corrects expressive lengthening / word lengthening (hellooo --> hello)
    posts = re.sub(r'(.)\1{3,}', r'\1', posts)
    # print("correct expressive length: \n" + posts)

    # Tokenization: ["Hello my name is"] --> ["Hello", "my", "name", "is"]
    tokens = nltk.word_tokenize(posts.lower())
    # print("create tokens: \n" + str(tokens))


    # remove stopwords i.e. "and"
    stopwords_ = set(stopwords.words("english"))

    # https://escholarship.org/content/qt6n5652cx/qt6n5652cx.pdf
    paper_words = ["got", "a", "i", "il", "be", "the", "of", "do", "not", "can", "am"]

    custom_stopwords = stopwords_ - set(paper_words)

    clean_tokens = []
    for token in tokens:
        if token not in custom_stopwords:
            clean_tokens.append(token)
    # print("stopwords removed tokens: \n" + str(clean_tokens))

    # Keep context of word i.e. changing --> change (instead of changing --> chang, when using stemming)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in tokens:
        # print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))
        lemmas.append(lemmatizer.lemmatize(w))
    # print("lemmatized text corpus: \n" + str(lemmas))

    # concat lemmas back to text
    clean_text = " ".join(lemmas)

    return clean_text


# Data Exploration
def exploration(df, n):

    """
    :param df: rows of preprocessed text column from dataset
    :param n: top N-frequent words
    :return: top N-words frequency graph and wordcloud of preprocessed text and plot distribution of dataset
    """

    preprocessed_text = df["preprocessed_text"][0:N]

    for i in preprocessed_text:
        corpus = "".join(i)

    tokens = nltk.word_tokenize(corpus.lower())
    # print(tokens)

    frequency_distribution = FreqDist(tokens)
    plt.ion()
    frequency_distribution.plot(n, title="Most " + str(n) + "-Words in corpus", cumulative=False)
    plt.savefig("img/" + "word_frequency.png")
    plt.ioff()
    plt.show()
    plt.ion()

    word_cloud = WordCloud(collocations=False, background_color='white').generate(corpus)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("img/" + "corpus_wordcloud.png")
    plt.ioff()
    plt.show()

    # Plot distribution of dataset - (un)balanced?
    fig, ax = plt.subplots()
    fig.suptitle("Dataset Distribution", fontsize=12)
    df.reset_index().groupby("type").count().sort_values(by=
                                                        "index").plot(kind="barh", legend=False,
                                                                      ax=ax).grid(axis='x')
    plt.savefig("img/" + "dataset_distribution.png")
    plt.ioff()
    plt.show()


# MBTI Personality Types in different CSV files
def group_data(df):
    """
    :param df: (preprocessed) data
    :return: create csv file for every MBTI type
    """

    df_grouped = df.groupby(by="type")

    for perso_type, data in df_grouped:

        if perso_type == "ENFJ":
            df_ENFJ = pd.DataFrame(data)
            df_ENFJ.to_csv("data/personality_types/df_ENFJ.csv")
            # print(perso_type)
            # print(df_ENFJ)
            # print(data["posts"])

        elif perso_type == "ENFP":
            df_ENFP = pd.DataFrame(data)
            df_ENFP.to_csv("data/personality_types/df_ENFP.csv")
            # print(perso_type)
            # print(df_ENFP)

        elif perso_type == "ENTJ":
            df_ENTJ = pd.DataFrame(data)

            df_ENTJ.to_csv("data/personality_types/df_ENTJ.csv")
            # print(perso_type)
            # print(df_ENTJ)

        elif perso_type == "ENTP":
            df_ENTP = pd.DataFrame(data)
            df_ENTP.to_csv("data/personality_types/df_ENTP.csv")
            # print(perso_type)
            # print(df_ENTP)

        elif perso_type == "ESFJ":
            df_ESFJ = pd.DataFrame(data)
            df_ESFJ.to_csv("data/personality_types/df_ESFJ.csv")
            # print(perso_type)
            # print(df_ESFJ)

        elif perso_type == "ESFP":
            df_ESFP = pd.DataFrame(data)
            df_ESFP.to_csv("data/personality_types/df_ESFP.csv")
            # print(perso_type)
            # print(df_ESFP)

        elif perso_type == "ESTP":
            df_ESTP = pd.DataFrame(data)
            df_ESTP.to_csv("data/personality_types/df_ESTP.csv")
            # print(perso_type)
            # print(df_ESTP)

        elif perso_type == "INFJ":
            df_INFJ = pd.DataFrame(data)
            df_INFJ.to_csv("data/personality_types/df_INFJ.csv")
            # print(perso_type)
            # print(df_INFJ)

        elif perso_type == "INFP":
            df_INFP = pd.DataFrame(data)
            df_INFP.to_csv("data/personality_types/df_INFP.csv")
            # print(perso_type)
            # print(df_INFP)

        elif perso_type == "INTJ":
            df_INTJ = pd.DataFrame(data)
            df_INTJ.to_csv("data/personality_types/df_INTJ.csv")
            # print(perso_type)
            # print(df_INTJ)

        elif perso_type == "INTP":
            df_INTP = pd.DataFrame(data)
            df_INTP.to_csv("data/personality_types/df_INTP.csv")
            # print(perso_type)
            # print(df_INTP)

        elif perso_type == "ISFJ":
            df_ISFJ = pd.DataFrame(data)
            df_ISFJ.to_csv("data/personality_types/df_ISFJ.csv")
            # print(perso_type)
            # print(df_ISFJ)

        elif perso_type == "ISFP":
            df_ISFP = pd.DataFrame(data)
            df_ISFP.to_csv("data/personality_types/df_ISFP.csv")
            # print(perso_type)
            # print(df_ISFP)

        elif perso_type == "ISTJ":
            df_ISTJ = pd.DataFrame(data)
            df_ISTJ.to_csv("data/personality_types/df_ISTJ.csv")
            # print(perso_type)
            # print(df_ISTJ)

        elif perso_type == "ISTP":
            df_ISTP = pd.DataFrame(data)
            df_ISTP.to_csv("data/personality_types/df_ISTP.csv")
            # print(perso_type)
            # print(df_ISTP)


# Create dataset with binary labels
def binary_labeling(list):

    """
    :param: List of csv files (personality types)
    :return: one big csv file with summarized extroverted and introverted types
    Extroverted - 1
    Introverted - 0
    """

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
        df_binary_all.to_csv("data/df_binary_preprocessed.csv")



# ---------------------------------------------------------------------------------------------------------------------

# Initialize

# Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
N = 1000 # len of dataset 8675

# import raw dataset
df = pd.read_csv("data/mbti_1.csv")
print("Raw Dataset")
print(df.head())


# create new csv file of dataset with added preprocessed text data
df["preprocessed_text"] = df["posts"][0:N].apply(lambda x: preprocessing(x))
df.to_csv("data/df_multi_preprocessed.csv")
print("Preprocessed Multiclassification Dataset")
print(df.head() + "\n")


# Data Exploration
exploration(df, 50)


# Create different MBTI personality groups and output them as csv files
# group_data(df)


# Label MBTI groups as extroverted (1) and introverted (0) in one big csv file
#list = os.listdir("data/personality_types")
#binary_labeling(list)
#df_binary = pd.read_csv("data/df_binary_preprocessed.csv")
#print("Preprocessed Binary Dataset")
#print(df_binary.head())
