import re
import tldextract


def convert_posts_to_list(post: str) -> list:
    return post.split("|||")


def avg_word_count(posts: list) -> float:
    return sum([len(post.split()) for post in posts]) / len(posts)


#Returns the avg count of 2 or more exclamation marks behind each other
def avg_exclamation_mark_count(posts: list) -> float:
    return sum([len(re.findall(r"!{2,}", post)) for post in posts]) / len(posts)


#Returns the avg count of 2 or more full stops behind each other
def avg_full_stop_count(posts: list) -> float:
    return sum([len(re.findall(r"\.{2,}", post)) for post in posts]) / len(posts)


def replace_url_with_domain(posts: list) -> str:
    
    for i, post in enumerate(posts):
        urls = re.findall(r"(https?://[^\s]+)", post)
        for url in urls:
            domain = tldextract.extract(url).domain
            post = post.replace(url, domain)
        posts[i] = post

    return posts


#remove trailing 3 dots from posts
def remove_trailing_3_dots(posts: list) -> str:
    
    # check if last 3 chars are "..."
    for i, post in enumerate(posts):
        #Posts in the dataset are cut off after 192 characters but not exactly, only after the next word is finished. The cut off is indicated by 3 dots at the end of the post.
        if len(post) >= 192 and post[-3:] == "...":
            posts[i] = post[:-3]

    return posts


def avg_count_of_hello(posts: list) -> float:
    return sum([len(re.findall(r"hello ", post)) for post in posts]) / len(posts)

def avg_count_of_hi(posts: list) -> float:
    return sum([len(re.findall(r"hi ", post)) for post in posts]) / len(posts)
    
def correct_expressive_lengthening(posts: list) -> list:
    for i, post in enumerate(posts):
        #correct expressive lengthening

        post = re.sub(r"([a-zA-Z])\1{2,}", r"\1", post)
        posts[i] = post

    return posts

#remove non alphabetical characters
def remove_non_alphabetical_characters(posts: list) -> list:
    for i, post in enumerate(posts):
        post = re.sub(r"[^a-zA-Z ]", "", post)
        posts[i] = post

    return posts



#average count of extroverted bigrams in posts
def avg_count_of_extroverted_bigrams(posts: list) -> float:
    return sum([sum([post.count(bigram) for bigram in extroverted_bigrams]) for post in posts]) / len(posts)

def avg_count_of_extroverted_stylistic_impressions(posts: list) -> float:
    return sum([sum([post.count(stylistic_impression) for stylistic_impression in extroverted_stylistic_impressions]) for post in posts]) / len(posts)

#average count of interoverted quantifiers
def avg_count_of_interoverted_quantifiers(posts: list) -> float:
    return sum([sum([post.count(quantifier) for quantifier in interoverted_quantifiers]) for post in posts]) / len(posts)

#average count of introverted first person singular pronoun
def avg_count_of_introverted_first_person_singular_pronoun(posts: list) -> float:
    return sum([sum([post.count(pronoun) for pronoun in introverted_first_person_singular_pronoun]) for post in posts]) / len(posts)

#average count of introverted negations
def avg_count_of_introverted_negations(posts: list) -> float:
    return sum([sum([post.count(negation) for negation in introverted_negations]) for post in posts]) / len(posts)

def avg_count_of_emojis(posts: list) -> float:
    return sum([len(re.findall(r':(.*?):', post)) for post in posts]) / len(posts)
