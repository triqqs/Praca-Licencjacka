import pandas as pd
import spacy as sp
import string

nlp = sp.load("pl_core_news_lg")  # wczytanie polskiej bazy

data = pd.read_csv("./dane/nawl_val_aro.csv", delimiter=";")
nawl_words = data.NAWL_word  # wczytanie słów z bazy NAWL

argumentacje = pd.read_csv("./dane/Zadanie_październik_przesłanki.csv", delimiter=";", encoding="cp1250",
                           usecols=["Konkluzja", "Przesłanka 1", "Przesłanka 2", "Przesłanka 3", "Przesłanka 4", "Przesłanka 5"])


konk = argumentacje["Konkluzja"].values
p1 = argumentacje["Przesłanka 1"].values
p2 = argumentacje["Przesłanka 2"].values
p3 = argumentacje["Przesłanka 3"].values
p4 = argumentacje["Przesłanka 4"].values
p5 = argumentacje["Przesłanka 5"].values


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


nawl_match = list_matcher(konk, nawl_words)
print(nawl_match)
nawl_match = list_matcher(p1, nawl_words)
print(nawl_match)
nawl_match = list_matcher(p2, nawl_words)
print(nawl_match)
nawl_match = list_matcher(p3, nawl_words)
print(nawl_match)
nawl_match = list_matcher(p4, nawl_words)
print(nawl_match)
nawl_match = list_matcher(p5, nawl_words)
print(nawl_match)
