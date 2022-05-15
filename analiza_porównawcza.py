import pandas as pd
from analiza_nawl import argumentacje_df as nawl_df
from analiza_anotacji import argumentacje_z_anotacja_df as anotacja_df

import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import plotly.express as px
from numpy import nan


#tworzenie kolumny z nowymi wartościami skategoryzowanymi dla anotacji
def anotacja(row):
    if row['pos'] == 1:
        val = 1
    elif row['neg'] == 1:
        val = -1
    else:
        val = 0
    return val


def kappa(df):
    tp = df.loc[1, 1]
    tn = df.loc[-1, -1]
    fp = df.loc[-1, 1]
    fn = df.loc[1, -1]
    return (2*(tp*tn-fn*fp))/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))


# Przy pie-chart specs type musi być domain
specs = [[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]
subplot_titles = ('NAWL', 'Anotacja',
                  'NAWL z pominięciem wartości nieokreślonych',
                  'Anotacja z pominięciem wartości które<br>zostały nieokreślone w bazie NAWL')

for (pos_column, neg_column) in [('Pos_sentiment_K', 'Neg_sentiment_K'), ('Pos_sentiment_1', 'Neg_sentiment_1'),
                                 ('Pos_sentiment_2', 'Neg_sentiment_2'), ('Pos_sentiment_3', 'Neg_sentiment_3')]:
    fig = make_subplots(rows=2, cols=2, specs=specs,
                        horizontal_spacing=0.05, subplot_titles=subplot_titles)

    if pos_column[-1] == 'K':
        nawl_column = f"k-cat"
        anot_column = f"k-anot"
    else:
        nawl_column = f"p{pos_column[-1]}-cat"
        anot_column = f"p{pos_column[-1]}-anot"

    # liczenie wystąpień -1/0/1 oraz konwersja do formatu który jest wygodny do robienia wykresu
    #cts (counts) => df liczby wystąpień wartości skategoryzowanych
    cts = nawl_df[nawl_column].value_counts().to_frame()

    labels_dict = {1: 'Pozytywne', -1: 'Negatywne', 0: 'Nieokreślone'}

    fig.add_trace(
        go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
               textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
               marker_colors=('#4D96FF', '#6BCB77', '#FF6B6B')
               ),
        row=1, col=1
    )

    #df zawierający wartości skategoryzowane z anotacja df (z dwóch kolumn w jedną)
    anot_df = pd.DataFrame(data={'pos': anotacja_df[pos_column], 'neg': anotacja_df[neg_column]})
    anot_df[anot_column] = anot_df.apply(anotacja, axis=1)
    cts = anot_df[anot_column].value_counts().to_frame()

    fig.add_trace(
        go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
               textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
               marker_colors=('#6BCB77', '#FF6B6B')
               ),
        row=1, col=2
    )

    #usuwanie 0 (wartości nieokreślonych) z nawl_df
    without_zeros = nawl_df.replace(0, nan)
    cts = without_zeros[nawl_column].value_counts().to_frame()
    fig.add_trace(
        go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
               textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
               marker_colors=('#6BCB77', '#FF6B6B')
               ),
        row=2, col=1
    )

    #łączenie anot_df i nawl_df w jednego df (merged_df)
    merged_df = pd.concat([anot_df[anot_column], nawl_df[nawl_column].reindex(anot_df.index)], axis=1)
    #usuwanie z merged_df ADU (argumentative discurse unikt) lub całych argumentów które były nieokreślone dla bazy nawl
    merged_wo0 = merged_df.drop(merged_df[merged_df[nawl_column] == 0].index)
    cts = merged_wo0[anot_column].value_counts().to_frame()
    fig.add_trace(
        go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
               textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
               marker_colors=('#6BCB77', '#FF6B6B')
               ),
        row=2, col=2
    )
    fig.update_layout(title_text=pos_column)
    fig.show()

    # TODO DO PRZEJRZENIA, POTEM USUNĄĆ Z PĘTLI (usunąć wycięcie)
    #tworzenie wizualizacji confusion matrix (tablicy pomyłek) (dla poszczególnych ADU)
    hm_palette = sns.cubehelix_palette(as_cmap=True, reverse=True)
    merged_df = pd.concat([nawl_df, anotacja_df.reindex(nawl_df.index)], axis=1)
    merged_wo0 = merged_df.drop(merged_df[merged_df[nawl_column] == 0].index)

    merged_wo0.rename(columns={pos_column: 'pos', neg_column: 'neg'}, inplace=True)
    merged_wo0[anot_column] = merged_wo0.apply(anotacja, axis=1)
    data = {'nawl': merged_wo0[nawl_column],
            'anotacja': merged_wo0[anot_column]}

    data_df = pd.DataFrame(data, columns=['nawl', 'anotacja'])
    confusion_matrix = pd.crosstab(data_df['anotacja'], data_df['nawl'], rownames=['Anotacja'], colnames=['NAWL'])
    confusion_df = confusion_matrix.unstack().reorder_levels(('Anotacja', 'NAWL'))
    print(f"Kappa: {round(kappa(confusion_df), 3)}")
    ax = sns.heatmap(confusion_matrix, annot=True, cmap=hm_palette, fmt='.3g')
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()


# Tworzenie confusion matrix dla całości argumentów z bazy (K,1,2,3)
merged_df = pd.concat([nawl_df, anotacja_df.reindex(nawl_df.index)], axis=1)
merged_wo0 = merged_df.drop(merged_df[merged_df['k-cat'] == 0].index)
merged_wo0.rename(columns={'Pos_sentiment_K': 'pos', 'Neg_sentiment_K': 'neg'}, inplace=True)
merged_wo0['k-anot'] = merged_wo0.apply(anotacja, axis=1)

data = {'nawl': merged_wo0['k-cat'], 'anotacja': merged_wo0['k-anot']}
all_comparison_df = pd.DataFrame(data, columns=['nawl', 'anotacja'])
all_comparison_w0_df = pd.DataFrame(data, columns=['nawl', 'anotacja'])
for (pos_column, neg_column) in [('Pos_sentiment_1', 'Neg_sentiment_1'), ('Pos_sentiment_2', 'Neg_sentiment_2'),
                                 ('Pos_sentiment_3', 'Neg_sentiment_3')]:
    nawl_column = f"p{pos_column[-1]}-cat"
    anot_column = f"p{pos_column[-1]}-anot"
    merged_wo0 = merged_df.drop(merged_df[merged_df[nawl_column] == 0].index)
    merged_wo0 = merged_wo0.dropna(subset=[f"Przesłanka {pos_column[-1]}"])
    merged_wo0.rename(columns={pos_column: 'pos', neg_column: 'neg'}, inplace=True)

    merged_wo0[anot_column] = merged_wo0.apply(anotacja, axis=1)

    to_be_added = pd.DataFrame({'nawl': merged_wo0[nawl_column],
                                'anotacja': merged_wo0[anot_column]},
                               columns=['nawl', 'anotacja'])
    to_be_added_w0 = pd.DataFrame({'nawl': merged_df[nawl_column],
                                   'anotacja': merged_df[anot_column]},
                                  columns=['nawl', 'anotacja'])
    # print(merged_wo0[[f"Przesłanka {pos_column[-1]}", nawl_column, anot_column, 'pos', 'neg']].head(7))
    # print(merged_wo0[[nawl_column, anot_column]])
    all_comparison_df = pd.concat([all_comparison_df, to_be_added], ignore_index=True)
    all_comparison_w0_df = pd.concat([all_comparison_w0_df, to_be_added_w0], ignore_index=True)
    # print(to_be_added.shape, all_comparison_df.shape)

# # pie chart dla wszystkich argumentów
# merged_df = pd.concat([anot_df[anot_column], nawl_df[nawl_column].reindex(anot_df.index)], axis=1)
# #usuwanie z merged_df ADU (argumentative discurse unikt) lub całych argumentów które były nieokreślone dla bazy nawl
# merged_wo0 = merged_df.drop(merged_df[merged_df[nawl_column] == 0].index)
cts = all_comparison_w0_df['nawl'].value_counts().to_frame()
fig.add_trace(
    go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
           textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
           marker_colors=('#4D96FF', '#6BCB77', '#FF6B6B')
           ),
    row=1, col=1
)
cts = all_comparison_w0_df['anotacja'].value_counts().to_frame()
fig.add_trace(
    go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
           textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
           marker_colors=('#6BCB77', '#FF6B6B')
           ),
    row=1, col=2
)
cts = all_comparison_df['nawl'].value_counts().to_frame()
fig.add_trace(
    go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
           textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
           marker_colors=('#4D96FF', '#6BCB77', '#FF6B6B')
           ),
    row=2, col=1
)
cts = all_comparison_df['anotacja'].value_counts().to_frame()
fig.add_trace(
    go.Pie(labels=[labels_dict[i] for i in cts.index.tolist()], values=[value[0] for value in cts.values.tolist()],
           textposition='inside', textinfo='label+percent+value', insidetextorientation='horizontal',
           marker_colors=('#6BCB77', '#FF6B6B')
           ),
    row=2, col=2
)
fig.update_layout(title_text='Pie chart dla wszystkich argumentów')
fig.show()

# print(all_comparison_df)
confusion_matrix = pd.crosstab(all_comparison_df['anotacja'], all_comparison_df['nawl'],
                               rownames=['Anotacja'], colnames=['NAWL'])
confusion_df = confusion_matrix.unstack().reorder_levels(('Anotacja', 'NAWL'))
print(f"Kappa: {round(kappa(confusion_df), 3)}")
ax = sns.heatmap(confusion_matrix, annot=True, cmap=hm_palette, fmt='.3g')
ax.invert_yaxis()
ax.invert_xaxis()
plt.show()
