import pandas as pd

argumentacje_z_anotacja_df = pd.read_csv("./dane/Baza argumentacji dotyczącej zdrowia - anotacja.csv", delimiter=",",
                           usecols=["Konkluzja", "Pos_sentiment_K", "Neg_sentiment_K", "Przesłanka 1", "Pos_sentiment_1",
                                    "Neg_sentiment_1", "Przesłanka 2", "Pos_sentiment_2", "Neg_sentiment_2",
                                    "Przesłanka 3", "Pos_sentiment_3", "Neg_sentiment_3", "Przesłanka 4", "Pos_sentiment_4",
                                    "Neg_sentiment_4", "Przesłanka 5", "Pos_sentiment_5", "Neg_sentiment_5", "Przesłanka 6",
                                    "Pos_sentiment_6", "Neg_sentiment_6", "Przesłanka 7", "Pos_sentiment_7", "Neg_sentiment_7",
                                    "Przesłanka 8", "Pos_sentiment_8", "Neg_sentiment_8"])  # usunąć fillna aby zobaczyć ilość przesłanek


if __name__ == "__main__":
    #print(pos_sen_K.mean())
    #print(neg_sen_K.mean())
    print(argumentacje_z_anotacja_df.info())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(argumentacje_z_anotacja_df[["Przesłanka 3", "Pos_sentiment_3"]])




