import pandas as pd
import spacy as sp
import string
from statistics import mean

nlp = sp.load("pl_core_news_lg")  # wczytanie polskiej bazy

data = pd.read_csv("./dane/nawl_val_aro.csv", delimiter=";", decimal=',')
data['val_M_all'] = data['val_M_all'].astype(float)

nawl_words = data.NAWL_word  # wczytanie słów z bazy NAWL

argumentacje = pd.read_csv("./dane/Baza_arg_dot_zdrowia.csv", delimiter=",",
                           usecols=["Konkluzja", "Przesłanka 1", "Przesłanka 2", "Przesłanka 3", "Przesłanka 4",
                                    "Przesłanka 5", "Przesłanka 6", "Przesłanka 7", "Przesłanka 8"])

konk = argumentacje["Konkluzja"].values
p1 = argumentacje["Przesłanka 1"].values
p2 = argumentacje["Przesłanka 2"].values
p3 = argumentacje["Przesłanka 3"].values
p4 = argumentacje["Przesłanka 4"].values
p5 = argumentacje["Przesłanka 5"].values
p6 = argumentacje["Przesłanka 6"].values
p7 = argumentacje["Przesłanka 7"].values
p8 = argumentacje["Przesłanka 8"].values


def my_matcher(mylist, emolist):  # zwraca część spólną dwóch list (bez powtórzeń)
    return set(mylist).intersection(set(emolist))


def tokenizer(my_string):  # dzieli stringa na listę podzieloną spacjami
    my_string = my_string.translate(str.maketrans('', '', string.punctuation))  # usuwa znaki interpunkcyjne
    return my_string.split(" ")


def lemmatizer(mylist):  # zwraca wyrazy zmienione do formy podstawowej
    doc = nlp(" ".join(mylist))
    my_lemma = []
    for token in doc:
        my_lemma += [token.lemma_]
    return my_lemma


def list_matcher(my_lists, emo_list):
    result = []
    for l in my_lists:
        if type(l) == float:
            continue
        l = lemmatizer(tokenizer(l))
        result += [my_matcher(l, emo_list)]
    return result


def word_valence(my_word, emo_df):
    return emo_df.loc[emo_df['NAWL_word'] == my_word, 'val_M_all'].iloc[0]


def sentence_valence(my_sentence, emo_df):
    if len(my_sentence) == 0:
        return None
    result = []
    for word in my_sentence:
        result += [word_valence(word, emo_df)]
    return mean(result)


def print_matched_sentences(sentences_list, emo_df):
    nawl_match = list_matcher(sentences_list, emo_df.NAWL_word)
    valences = []
    for sentence in nawl_match:
        result = sentence_valence(sentence, data)
        if result:
            valences += [result]
        print(result, end='\t')
    print()
    print(nawl_match)
    print(mean(valences))
    print()


print_matched_sentences(konk, data)
print_matched_sentences(p1, data)
print_matched_sentences(p2, data)
print_matched_sentences(p3, data)
print_matched_sentences(p4, data)
print_matched_sentences(p5, data)
print_matched_sentences(p6, data)
print_matched_sentences(p7, data)
print_matched_sentences(p8, data)

