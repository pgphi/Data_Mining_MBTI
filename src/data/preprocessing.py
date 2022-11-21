import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import pandas as pd
import random
import re
import tldextract
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def tokenization(text: str):
    return pd.Series(nltk.word_tokenize(text.lower()))


def removal(tokens: pd.Series):

    stopwords_list = stopwords.words("english")

    tokens = tokens.apply(lambda token: token.translate(str.maketrans('', '', string.punctuation)))
    tokens = tokens.apply(lambda token: token if token not in stopwords_list and token != '' else None).dropna()

    return tokens


def stemming(tokens: pd.Series):

    stemmer = PorterStemmer()

    return tokens.apply(lambda token: stemmer.stem(token))


def lemmatization(tokens: pd.Series):

    lemmatizer = WordNetLemmatizer()

    return tokens.apply(lambda token: lemmatizer.lemmatize(token))


def query_expansion(tokens: pd.Series, sample_size=2):

    token_list = tokens.tolist()

    new_tokenlist = []
    for token in token_list:
        synonyms = get_synonyms(token, sample_size)

        new_tokenlist.append(token)
        if len(synonyms) > 0:
            new_tokenlist.extend(synonyms)

    return pd.Series(new_tokenlist)


def get_synonyms(phrase, sample_size):

    synonyms = []
    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            if '_' not in l.name() and l.name() != phrase:
                synonyms.append(l.name())

    synonym_set = set(synonyms)

    if sample_size > len(synonym_set):
        return list(synonym_set)
    else:
        synonym_set_sampled = random.sample(synonym_set, sample_size)
        return list(synonym_set_sampled)


#Remove non-alpha characters and preservers whitespaces (me420 i'm done that's it!!! --> me im done thats it)

def split_posts(posts: str) -> str:
    return posts.replace("|||", " ")

    
def remove_non_alpha_words(posts: str) -> str:
    
    #Removes non alphabet chars and \s for matching whitespace
    regex = re.compile('[^a-zA-Z\s]')
    posts = regex.sub('', posts)
    return posts


#regex match 2 colons : around a string and replace with just the string
def handle_emojis(posts: str) -> str:
    return re.sub(r':(.*?):', r'\1' + "emoji", posts)


#Replaces URLs with Second-Level-Domain (https://www.youtube.com/watch?v=dQw4w9WgXcQ --> youtube)
def replace_url_with_domain(posts: str) -> str:
    
    url_list = url_list = re.findall(r'(https?://[^\s]+)', posts)

    if len(url_list) > 0:
        for url in url_list:
            domain = tldextract.extract(url).domain
            posts = posts.replace(url, domain)

    return posts


#Corrects expressive lengthening / word lengthening (hellooo --> hello)
def correct_expressive_lengthening(posts: str) -> str:
    
    return re.sub(r'(.)\1{3,}', r'\1', posts)


def tokenize_posts(posts: str) -> list[str]:
    tokens = nltk.word_tokenize(posts.lower())
    return tokens


def remove_stopwords(tokens):
    stopwords_ = set(stopwords.words("english"))

    #https://escholarship.org/content/qt6n5652cx/qt6n5652cx.pdf
    paper_words = ["got", "a", "i", "il", "be", "the", "of", "do", "not", "can", "am"]

    custom_stopwords = stopwords_ - set(paper_words)

    clean_tokens = []
    for token in tokens:
        if token not in custom_stopwords:
            clean_tokens.append(token)

    return clean_tokens
    
    
def lemmatize_posts(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in tokens:
        # print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))
        lemmas.append(lemmatizer.lemmatize(w))
    return lemmas



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
    :return: preprocessed csv file for each personality type and evaluations for each type i.e. ENFJ and numeric labels:
    """

    df_grouped = df.groupby(by="type")

    for perso_type, data in df_grouped:

        if perso_type == "ENFJ":
            df_ENFJ = pd.DataFrame(data)
            df_ENFJ.to_csv("data/personality_types/df_ENFJ.csv")
            print(perso_type)
            print(df_ENFJ)
            print(data["posts"])
            preprocessing(df_ENFJ, "ENFJ")

        elif perso_type == "ENFP":
            df_ENFP = pd.DataFrame(data)
            df_ENFP.to_csv("data/personality_types/df_ENFP.csv")
            print(perso_type)
            print(df_ENFP)
            preprocessing(df_ENFP, "ENFP")

        elif perso_type == "ENTJ":
            df_ENTJ = pd.DataFrame(data)

            df_ENTJ.to_csv("data/personality_types/df_ENTJ.csv")
            print(perso_type)
            print(df_ENTJ)
            preprocessing(df_ENTJ, "ENTJ")

        elif perso_type == "ENTP":
            df_ENTP = pd.DataFrame(data)
            df_ENTP.to_csv("data/personality_types/df_ENTP.csv")
            print(perso_type)
            print(df_ENTP)
            preprocessing(df_ENTP, "ENTP")

        elif perso_type == "ESFJ":
            df_ESFJ = pd.DataFrame(data)
            df_ESFJ.to_csv("data/personality_types/df_ESFJ.csv")
            print(perso_type)
            print(df_ESFJ)
            preprocessing(df_ESFJ, "ESFJ")

        elif perso_type == "ESFP":
            df_ESFP = pd.DataFrame(data)
            df_ESFP.to_csv("data/personality_types/df_ESFP.csv")
            print(perso_type)
            print(df_ESFP)
            preprocessing(df_ESFP, "ESFP")

        elif perso_type == "ESTP":
            df_ESTP = pd.DataFrame(data)
            df_ESTP.to_csv("data/personality_types/df_ESTP.csv")
            print(perso_type)
            print(df_ESTP)
            preprocessing(df_ESTP, "ESTP")

        elif perso_type == "INFJ":
            df_INFJ = pd.DataFrame(data)
            df_INFJ.to_csv("data/personality_types/df_INFJ.csv")
            print(perso_type)
            print(df_INFJ)
            preprocessing(df_INFJ, "INFJ")

        elif perso_type == "INFP":
            df_INFP = pd.DataFrame(data)
            df_INFP.to_csv("data/personality_types/df_INFP.csv")
            print(perso_type)
            print(df_INFP)
            preprocessing(df_INFP, "INFP")

        elif perso_type == "INTJ":
            df_INTJ = pd.DataFrame(data)
            df_INTJ.to_csv("data/personality_types/df_INTJ.csv")
            print(perso_type)
            print(df_INTJ)
            preprocessing(df_INTJ, "INTJ")

        elif perso_type == "INTP":
            df_INTP = pd.DataFrame(data)
            df_INTP.to_csv("data/personality_types/df_INTP.csv")
            print(perso_type)
            print(df_INTP)
            preprocessing(df_INTP, "INTP")

        elif perso_type == "ISFJ":
            df_ISFJ = pd.DataFrame(data)
            df_ISFJ.to_csv("data/personality_types/df_ISFJ.csv")
            print(perso_type)
            print(df_ISFJ)
            preprocessing(df_ISFJ, "ISFJ")

        elif perso_type == "ISFP":
            df_ISFP = pd.DataFrame(data)
            df_ISFP.to_csv("data/personality_types/df_ISFP.csv")
            print(perso_type)
            print(df_ISFP)
            preprocessing(df_ISFP, "ISFP")

        elif perso_type == "ISTJ":
            df_ISTJ = pd.DataFrame(data)
            df_ISTJ.to_csv("data/personality_types/df_ISTJ.csv")
            print(perso_type)
            print(df_ISTJ)
            preprocessing(df_ISTJ, "ISTJ")

        elif perso_type == "ISTP":
            df_ISTP = pd.DataFrame(data)
            df_ISTP.to_csv("data/personality_types/df_ISTP.csv")
            print(perso_type)
            print(df_ISTP)
            preprocessing(df_ISTP, "ISTP")


# load data
# DF = pd.read_csv("data/mbti_1.csv")

# grouping and preprocessing
# group_prepro_data(DF)

# create dataset with binary and multi labels
def label_dataset(csv_files):

    """
    :return:
    ENFJ - 0
    ENFP - 1
    ENTJ - 3
    ENTP - 4
    ESFJ - 5
    ESFP - 6
    ESTP - 7
    INFJ - 8
    INFP - 9
    INTJ - 10
    INTP - 11
    ISFJ - 12
    ISFP - 13
    ISTJ - 14
    ISTP - 15
    ---------------
    Extroverted - 1
    Introverted - 0
    """

    list = csv_files

    df_introverted_binary = pd.DataFrame()
    df_extroverted_binary = pd.DataFrame()
    df_extroverted_multi = pd.DataFrame()
    df_introverted_multi = pd.DataFrame()

    for csv in list:

        if csv == "df_ENFJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENFJ", value=1, inplace=True)

        elif csv == "df_ENFP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENFP", value=1, inplace=True)

        elif csv == "df_ENTJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENTJ", value=1, inplace=True)

        elif csv == "df_ENTP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ENTP", value=1, inplace=True)

        elif csv == "df_ESFJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESFJ", value=1, inplace=True)

        elif csv == "df_ESFP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_extroverted_binary = df_extroverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESFP", value=1, inplace=True)

        elif csv == "df_ESTP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_extroverted_binary.replace(to_replace="ESTP", value=1, inplace=True)

        elif csv == "df_INFJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INFJ", value=0, inplace=True)

        elif csv == "df_INFP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INFP", value=0, inplace=True)

        elif csv == "df_INTJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INTJ", value=0, inplace=True)

        elif csv == "df_INTP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="INTP", value=0, inplace=True)

        elif csv == "df_ISFJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISFJ", value=0, inplace=True)

        elif csv == "df_ISFP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISFP", value=0, inplace=True)

        elif csv == "df_ISTJ.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISTJ", value=0, inplace=True)

        elif csv == "df_ISTP.csv":
            buffer = pd.read_csv("data/personality_types/"+csv)
            df_introverted_binary = df_introverted_binary.append(buffer, ignore_index=True)
            df_introverted_binary.replace(to_replace="ISTP", value=0, inplace=True)

        elif csv == "df_ENFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ENFJ", value=0, inplace=True)

        elif csv == "df_ENFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ENFP", value=1, inplace=True)

        elif csv == "df_ENTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ENTJ", value=2, inplace=True)

        elif csv == "df_ENTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ENTP", value=3, inplace=True)

        elif csv == "df_ESFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ESFJ", value=4, inplace=True)

        elif csv == "df_ESFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_extroverted_multi = df_extroverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ESFP", value=5, inplace=True)

        elif csv == "df_ESTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_extroverted_multi.replace(to_replace="ESTP", value=6, inplace=True)

        elif csv == "df_INFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="INFJ", value=7, inplace=True)

        elif csv == "df_INFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="INFP", value=8, inplace=True)

        elif csv == "df_INTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="INTJ", value=9, inplace=True)

        elif csv == "df_INTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="INTP", value=10, inplace=True)

        elif csv == "df_ISFJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="ISFJ", value=11, inplace=True)

        elif csv == "df_ISFP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="ISFP", value=12, inplace=True)

        elif csv == "df_ISTJ.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="ISTJ", value=13, inplace=True)

        elif csv == "df_ISTP.csv":
            buffer = pd.read_csv("data/personality_types/" + csv)
            df_introverted_multi = df_introverted_multi.append(buffer, ignore_index=True)
            df_introverted_multi.replace(to_replace="ISTP", value=14, inplace=True)


        # create csv files
        df_multi_all = df_introverted_multi.append(df_extroverted_multi)
        df_multi_all.to_csv("data/df_multi_all.csv")
        df_extroverted_multi.to_csv("data/df_extroverted_multi.csv")
        df_introverted_multi.to_csv("data/df_introverted_multi.csv")

        df_binary_all = df_introverted_binary.append(df_extroverted_binary)
        df_binary_all.to_csv("data/df_binary_all.csv")
        df_extroverted_binary.to_csv("data/df_extroverted_binary.csv")
        df_introverted_binary.to_csv("data/df_introverted_binary.csv")


    return [df_binary_all, df_introverted_binary, df_extroverted_binary,
            df_multi_all, df_introverted_multi, df_extroverted_multi]


# create csv datasets for classifier
#csv_files = os.listdir("data/personality_types")
#label_dataset(csv_files)


"""
df_bin_all = label_dataset(csv_files)[0]
#print(df_bin_all)
df_intro_bin = label_dataset(csv_files)[1]
#print(df_intro_bin)
df_extro_bin = label_dataset(csv_files)[2]
#print(df_extro_bin)
df_mult_all = label_dataset(csv_files)[3]
#print(df_mult_all)
df_extro_mult = label_dataset(csv_files)[4]
#print(df_extro_mult)
df_intro_mult = label_dataset(csv_files)[5]
#print(df_intro_mult)
"""




