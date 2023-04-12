import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt

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

def dia_maximo(df:pd.DataFrame):
    dia = []
    for sujeito in df.columns:
        max_pos = df.loc[df[sujeito] == max(df[sujeito])].index[0]
        dia_sem = floor(max_pos/1440) + 1
        dia.append(dia_sem)
    return dia

def maximos_por_sujeito(df):
    maximos = []
    for sujeito in df.columns:
        maximo = max(df[sujeito])
        maximos.append(maximo)
    
    return maximos



if __name__ == "__main__":
    df_data = pd.read_csv('data/timeseries_NEW.csv')
    df_groups = pd.read_csv('data/timeseries_classification.csv', index_col=0)

    df_data = df_data.drop(df_data.columns[0], axis=1)

    # Ler o arquivo .csv com os dados de série temporal
    df = pd.read_csv('data/timeseries_NEW.csv')

    """    # Definir o número de pontos em cada dia da semana (1440)
    n_points_per_day = 1440

    # Dividir a série temporal de cada sujeito em sub-séries temporais diárias
    subseries = df.values.reshape((-1, n_points_per_day, df.shape[1]))

    # Calcular a média de cada sub-série temporal diária
    means = np.mean(subseries, axis=1)

    # Criar um DataFrame com os resultados
    means_df = pd.DataFrame(means, columns=df.columns)



    # Agrupar os resultados por dia da semana e sujeito
    means_by_weekday = means_df.groupby(means_df.columns.weekday, axis=1).mean()"""

    amostras = 96

    fourier_amp = np.zeros((2, amostras))

    for j in range(2):
        for i in range(1, (amostras +1)):
            col_atual = df.iloc[:, i]
            fourier_amp[j][i-1] = abs(np.fft.fft(col_atual)[(j+1)*7])

    df2 = pd.DataFrame()

    df2["rolling5"]     = df[df.columns[0]].rolling(5).mean().values
    df2["rolling15"]    = df[df.columns[0]].rolling(15).mean().values
    df2["rolling60"]    = df[df.columns[0]].rolling(60).mean().values
    df2["fourieramp7"]  = fourier_amp[0][:]
    df2["fourieramp14"] = fourier_amp[1][:]


    print(fourier_amp)
    print(fourier_amp.shape)
    print(df2.head(20))

    max_fourier0 = df2['fourier0'].max()
    plt.plot(df2["fourier0"], 'r')
    print(max_fourier0)

    max_fourier1 = df2['fourier1'].max()
    plt.plot(df2["fourier1"], 'g')
    print(max_fourier1)

    max_fourier2 = df2['fourier2'].max()
    plt.plot(df2["fourier2"], 'b')
    print(max_fourier2)

    plt.show()