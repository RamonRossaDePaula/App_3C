import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_data = pd.read_csv('data/timeseries_NEW.csv')
df_groups = pd.read_csv('data/timeseries_classification.csv', index_col=0)

df_data = df_data.drop(df_data.columns[0], axis=1)

#print(df_data)
#print(df_groups)

def media_por_dia(df:pd.DataFrame):
    """ a ultima semana de todas as pessoas ta saindo igual a zero, achar o erro"""

    range_dias = np.array(range(0, len(df.axes[0])+1, 1440))
    print(range_dias)
    
    media = np.zeros((len(df.axes[1]), 7))

    for i in range(0, len(df.axes[1])):
        for j in range(0,7):
            media[i,j] = np.average(df.loc[range(range_dias[j], range_dias[j+1]-1),df.columns[i]])
    return media

print(media_por_dia(df_data))

