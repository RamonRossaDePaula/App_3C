import pandas as pd
import numpy as np

def medias(df:pd.DataFrame):
    media = np.zeros(len(df.axes[1]))
    for i in range(len(df.axes[1])):
        media[i] = np.average(df.loc[:,df.columns[i]])
    return media


def media_por_dia(df:pd.DataFrame):

    range_dias = np.array(range(0, len(df.axes[0])+1, 1440))
    media = np.zeros((len(df.axes[1]), 7))
    
    for i in range(0, len(df.axes[1])):
        for j in range(0,7):
            media[i,j] = np.average(df.loc[range(range_dias[j], range_dias[j+1]-1),df.columns[i]])
    return media


def vari_semanal(media:np.array, media_diaria:np.array) :

    var = np.zeros(len(media))
    
    for i in range(0, len(media)):
        for j in range(0,7):
            var[i] += ((media[i] - media_diaria[i, j])**2)/7
    return var


def maximos_por_sujeito(df):
    maximos = []
    for sujeito in df.columns:
        maximo = max(df[sujeito])
        max_pos = df.loc[df[sujeito] == maximo].index[0]
        maximos.append((maximo, max_pos))
    
    return maximos


if __name__ == "__main__":
    df_data = pd.read_csv('data/timeseries_NEW.csv')
    df_groups = pd.read_csv('data/timeseries_classification.csv', index_col=0)

    df_data = df_data.drop(df_data.columns[0], axis=1)

    media = medias(df_data)
    media_diaria = media_por_dia(df_data)

    print(maximos_por_sujeito(df_data))
