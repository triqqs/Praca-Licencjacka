import pandas as pd
from analiza_nawl import NAWL_value_list
from analiza_anotacji import baza_anotacji

import seaborn as sns
from matplotlib import pyplot as plt

baza_sum = pd.DataFrame({'Pochodzenie średniej walencji afektywnej': ['NAWL', 'Anotacja'],
                         'Walencja afektywna': [sum(NAWL_value_list)/3, sum(baza_anotacji)]
                         })
print(baza_sum)

colors = ['#31bdae', '#e592ae']
sns.set_theme(context='paper', style='darkgrid', palette=sns.color_palette(colors))
ax = sns.barplot(x='Pochodzenie średniej walencji afektywnej', y='Walencja afektywna', data=baza_sum, palette=sns.color_palette(colors))
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')  # oznaczenie dokładnej wartości walencji (dwa miejsca po przecinku)
plt.xticks(rotation=55)
plt.tight_layout()  # automatyczne dopasowanie marginesów - max wypełnienie figury

plt.show()


baza_porownawcza = pd.DataFrame({'Element argumentacji': ['Konkluzja', 'Przesłanka 1', 'Przesłanka 2', 'Przesłanka 3', 'Przesłanka 4',
                                           'Przesłanka 5', 'Przesłanka 6', 'Przesłanka 7', 'Przesłanka 8'],
                                'NAWL': NAWL_value_list, 'Anotacja': baza_anotacji})
baza_porownawcza['NAWL'] = baza_porownawcza['NAWL'] / 3

#tidy = baza_porownawcza.melt(id_vars='Nazwa').rename(columns=str.title)
tidy = baza_porownawcza.melt(id_vars='Element argumentacji', var_name='Pochodzenie', value_name='Walencja afektywna')
colors = ['#31bdae', '#e592ae']
sns.set_theme(context='paper', style='darkgrid', palette=sns.color_palette(colors))
ax = sns.barplot(x='Element argumentacji', y='Walencja afektywna', hue='Pochodzenie', data=tidy, palette=sns.color_palette(colors))
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')  # oznaczenie dokładnej wartości walenci (dwa miejsca po przecinku)
plt.xticks(rotation=55)
plt.tight_layout()  # automatyczne dopasowanie marginesów - max wypełnienie figury

plt.show()

