import pandas as pd
import spacy as sp
import string
from statistics import mean

nlp = sp.load("pl_core_news_lg")  # wczytanie polskiej bazy

data = pd.read_csv("./dane/nawl_val_aro.csv", delimiter=";", decimal=',')
data['val_M_all'] = data['val_M_all'].astype(float)

nawl_words = data.NAWL_word  # wczytanie słów z bazy NAWL

argumentacje_df = pd.read_csv("./dane/Baza argumentacji do metody leksykalnej.csv", delimiter=",",
                           usecols=["Konkluzja", "Przesłanka 1", "Przesłanka 2", "Przesłanka 3", "Przesłanka 4",
                                    "Przesłanka 5", "Przesłanka 6", "Przesłanka 7", "Przesłanka 8"])


def my_matcher(mylist, emolist):  # zwraca część wspólną dwóch list (bez powtórzeń)
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
            result += [False]  # dodajemy że komórka jest pusta
        else:
            l = lemmatizer(tokenizer(l))
            result += [my_matcher(l, emo_list)]
    return result


def word_valence(my_word, emo_df):
    return emo_df.loc[emo_df['NAWL_word'] == my_word, 'val_M_all'].iloc[0]


def sentence_valence(my_sentence, emo_df):
    if type(my_sentence) is bool:
        return None
    elif len(my_sentence) == 0:
        return 0
    result = []
    for word in my_sentence:
        result += [word_valence(word, emo_df)]
    return mean(result)


def all_sentences_valence(sentences_list: list, emo_df) -> list:
    nawl_match = list_matcher(sentences_list, emo_df.NAWL_word)
    valences = []
    for sentence in nawl_match:
        result = sentence_valence(sentence, data)
        sign = lambda x: x and (-1 if x < 0 else 1)
        valences += [sign(result) if result is not None else None]

        #print(result, end='\t')
    #print(nawl_match)
    return valences
    #print()


#tworzenie nowych kolumn ze skategoryzowanymi wartościami
argumentacje_df["k-cat"] = all_sentences_valence(argumentacje_df["Konkluzja"], data)
argumentacje_df["p1-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 1"], data)
argumentacje_df["p2-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 2"], data)
argumentacje_df["p3-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 3"], data)
argumentacje_df["p4-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 4"], data)
argumentacje_df["p5-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 5"], data)
argumentacje_df["p6-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 6"], data)
argumentacje_df["p7-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 7"], data)
argumentacje_df["p8-cat"] = all_sentences_valence(argumentacje_df["Przesłanka 8"], data)


if __name__ == "__main__":
    print(argumentacje_df['k-cat'].value_counts(dropna=False))
    print(argumentacje_df['p1-cat'].value_counts(dropna=False))
    print(argumentacje_df['p2-cat'].value_counts(dropna=False))
    print(argumentacje_df['p3-cat'].value_counts(dropna=False))
    print(argumentacje_df['p4-cat'].value_counts(dropna=False))
    print(argumentacje_df['p8-cat'].value_counts(dropna=False))
    # print(len(all_sentences_valence(argumentacje_df["Konkluzja"], data)))
    # print(len(all_sentences_valence(argumentacje_df["Przesłanka 5"], data)))
    # print(argumentacje_df["Konkluzja"])
