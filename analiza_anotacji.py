import pandas as pd
from statistics import mean
import numpy as np
import math

argumentacje_z_anotacja = pd.read_csv("./dane/Baza argumentacji dotyczącej zdrowia - anotacja.csv", delimiter=",",
                           usecols=["Konkluzja", "Pos_sentiment_K", "Neg_sentiment_K", "Przesłanka 1", "Pos_sentiment_1",
                                    "Neg_sentiment_1", "Przesłanka 2", "Pos_sentiment_2", "Neg_sentiment_2",
                                    "Przesłanka 3", "Pos_sentiment_3", "Neg_sentiment_3", "Przesłanka 4", "Pos_sentiment_4",
                                    "Neg_sentiment_4", "Przesłanka 5", "Pos_sentiment_5", "Neg_sentiment_5", "Przesłanka 6",
                                    "Pos_sentiment_6", "Neg_sentiment_6", "Przesłanka 7", "Pos_sentiment_7", "Neg_sentiment_7",
                                    "Przesłanka 8", "Pos_sentiment_8", "Neg_sentiment_8"]).fillna(0)


pos_sen_K = argumentacje_z_anotacja["Pos_sentiment_K"].values
neg_sen_K = argumentacje_z_anotacja["Neg_sentiment_K"].values
pos_sen_1 = argumentacje_z_anotacja["Pos_sentiment_1"].values
neg_sen_1 = argumentacje_z_anotacja["Neg_sentiment_1"].values
pos_sen_2 = argumentacje_z_anotacja["Pos_sentiment_2"].values
neg_sen_2 = argumentacje_z_anotacja["Neg_sentiment_2"].values
pos_sen_3 = argumentacje_z_anotacja["Pos_sentiment_3"].values
neg_sen_3 = argumentacje_z_anotacja["Neg_sentiment_3"].values
pos_sen_4 = argumentacje_z_anotacja["Pos_sentiment_4"].values
neg_sen_4 = argumentacje_z_anotacja["Neg_sentiment_4"].values
pos_sen_5 = argumentacje_z_anotacja["Pos_sentiment_5"].values
neg_sen_5 = argumentacje_z_anotacja["Neg_sentiment_5"].values
pos_sen_6 = argumentacje_z_anotacja["Pos_sentiment_6"].values
neg_sen_6 = argumentacje_z_anotacja["Neg_sentiment_6"].values
pos_sen_7 = argumentacje_z_anotacja["Pos_sentiment_7"].values
neg_sen_7 = argumentacje_z_anotacja["Neg_sentiment_7"].values
pos_sen_8 = argumentacje_z_anotacja["Pos_sentiment_8"].values
neg_sen_8 = argumentacje_z_anotacja["Neg_sentiment_8"].values

kon_value = pos_sen_K.mean() - neg_sen_K.mean()
p1_value = pos_sen_1.mean() - neg_sen_1.mean()
p2_value = pos_sen_2.mean() - neg_sen_2.mean()
p3_value = pos_sen_3.mean() - neg_sen_3.mean()
p4_value = pos_sen_4.mean() - neg_sen_4.mean()
p5_value = pos_sen_5.mean() - neg_sen_5.mean()
p6_value = pos_sen_6.mean() - neg_sen_6.mean()
p7_value = pos_sen_7.mean() - neg_sen_7.mean()
p8_value = pos_sen_8.mean() - neg_sen_8.mean()


baza_anotacji = [kon_value, p1_value, p2_value, p3_value, p4_value, p5_value, p6_value, p7_value, p8_value]

if __name__ == "__main__":
    #print(pos_sen_K.mean())
    #print(neg_sen_K.mean())
    print(kon_value)
    print(p1_value)
    print(p2_value)
    print(p3_value)
    print(p4_value)
    print(p5_value)
    print(p6_value)
    print(p7_value)
    print(p8_value)

    print(sum(baza_anotacji))

    #sumowanie anotacji dla danej kolumny
    #def sum_anotation(sentiment_list):
        #number_of_rows = sentiment_list.shape[0]
        #sentiment_mean = (sum([0 if x != x else x for x in sentiment_list]) / number_of_rows)
        #return sentiment_mean





