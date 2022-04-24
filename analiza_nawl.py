import pandas as pd
import spacy as sp
import string
from statistics import mean

nlp = sp.load("pl_core_news_lg")  # wczytanie polskiej bazy

data = pd.read_csv("./dane/nawl_val_aro.csv", delimiter=";", decimal=',')
data['val_M_all'] = data['val_M_all'].astype(float)

nawl_words = data.NAWL_word  # wczytanie słów z bazy NAWL

argumentacje = pd.read_csv("./dane/Baza argumentacji do metody leksykalnej.csv", delimiter=",",
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
        if type(l) == float:  # jeżeli lista nie jest listą tylko cyfrą/Null
            continue  # pomijamy
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


def all_sentences_valence(sentences_list: list, emo_df) -> float:
    nawl_match = list_matcher(sentences_list, emo_df.NAWL_word)
    valences = []
    for sentence in nawl_match:
        result = sentence_valence(sentence, data)
        if result:
            valences += [result]
        #print(result, end='\t')
    #print(nawl_match)
    return mean(valences)
    #print()

#zmiana walencji na zmienne kategorialne
def categorical_variable(list_of_valences):
    for walencja in list_of_valences:
        if walencja >= 0.2:
            pass


NAWL_value_list = [all_sentences_valence(konk, data), all_sentences_valence(p1, data), all_sentences_valence(p2, data),
                   all_sentences_valence(p3, data), all_sentences_valence(p4, data), all_sentences_valence(p5, data),
                   all_sentences_valence(p6, data), all_sentences_valence(p7, data), all_sentences_valence(p8, data)]


