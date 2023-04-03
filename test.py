import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.functions as fc

df_data = pd.read_csv('data/timeseries_NEW.csv')
df_groups = pd.read_csv('data/timeseries_classification.csv', index_col=0)

df_data = df_data.drop(df_data.columns[0], axis=1)

#print(df_data)
#print(df_groups)

media = fc.medias(df_data)
media_diaria = fc.media_por_dia(df_data)

def vari_semanal(media, media_diaria) :

    var = np.zeros(len(media))
    
    for i in range(0, len(media)):
        for j in range(0,7):
            var[i] += ((media[i] - media_diaria[i, j])**2)/7
    return var

print(vari_semanal(media, media_diaria))
print(vari_semanal(media, media_diaria).shape)


def plot_graph(df, sujeito, media):
    media_sujeito = media[0, 0]
    df = df[sujeito]
    dias = [[], [], [], [], [], [], []]
    j = 0
    for i in df:
        dias[j].append(i)
        j += 1
        if j == 7:
            j = 0

    segunda = dias[0]
    y = list(range(len(segunda)))

    print(np.var(segunda))
    print(np.mean(segunda))
    print(np.average(segunda))
    print(media_sujeito)

    # plt.scatter(y, segunda, color='crimson')
    # plt.hlines(y=media_sujeito, xmin=min(y), xmax=max(y), color='black')
    # plt.show()
    # print(media_sujeito)
    

plot_graph(df_data, df_data.columns[0], media)