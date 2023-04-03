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