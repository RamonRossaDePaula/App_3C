import pandas as pd
import numpy as np

def medias(df:pd.DataFrame):
    media = np.zeros(len(df.axes[1]))
    for i in range(len(df.axes[1])):
        media[i] = np.average(df.loc[:,df.columns[i]])
    return media

def media_por_semana(df:pd.DataFrame):
    """ a ultima semana de todas as pessoas ta saindo igual a zero, achar o erro"""

    range_semanas = np.array(range(0, len(df.axes[0])+1, 1440))
    media = np.zeros((len(df.axes[1]), 7))

    for i in range(0, len(df.axes[1])):
        for j in range(0,6):
            media[i,j] = np.average(df.loc[range(range_semanas[j], range_semanas[j+1]-1),df.columns[i]])
    return media