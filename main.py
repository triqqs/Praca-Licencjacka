import pandas as pd
import spacy as sp
import string

nlp = sp.load("pl_core_news_lg")  # wczytanie polskiej bazy

data = pd.read_csv("./dane/nawl_emo_cat.csv", delimiter=";")
nawl_words = data.NAWL_word  # wczytanie słów z bazy NAWL

my_words = ["wiosna", "miły", "chleb", "szczepionka", "złodziej", "pies", "absurd", "bukiet", "mama", "żałoba", "impreza", "miła"]
my_tekst = "Pies domowy – udomowiony gatunek ssaka drapieżnego z rodziny psowatych, traktowany przez niektóre ujęcia systematyczne za podgatunek wilka"


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


nawl_match = my_matcher(lemmatizer(my_words), nawl_words)
print(nawl_match)

nawl_match = my_matcher(lemmatizer(tokenizer(my_tekst)), nawl_words)
print(nawl_match)
