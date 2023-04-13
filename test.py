import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import src.functions as fc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#import nltk
import sklearn



df_data = pd.read_csv('data/timeseries_NEW.csv')
df_groups = pd.read_csv('data/timeseries_classification.csv', index_col=0)

df_data = df_data.drop(df_data.columns[0], axis=1)

#print(df_data)
#print(df_groups)

media = fc.medias(df_data)
media_diaria = fc.media_por_dia(df_data)

print(df_data.head())

print(df_data.shape)
def vari_semanal(media, media_diaria) :

    var = np.zeros(len(media))
    
    for i in range(0, len(media)):
        for j in range(0,7):
            var[i] += ((media[i] - media_diaria[i, j])**2)/7
    return var
amostras = 96

fourier_amp = np.zeros((2, amostras))

for j in range(2):
        for i in range(0, amostras):
            col_atual = df_data.iloc[:, [i]]
            fourier_amp[j][i] = abs(np.fft.fft(col_atual)[(j+1)*7])


predictors_df = pd.DataFrame()

predictors_df["mean"]           = fc.medias(df_data)
predictors_df["media_segunda"]  = fc.media_por_dia(df=df_data)[:, 0]
predictors_df["media_terca"]    = fc.media_por_dia(df=df_data)[:, 1]
predictors_df["media_quarta"]   = fc.media_por_dia(df=df_data)[:, 2]
predictors_df["media_quinta"]   = fc.media_por_dia(df=df_data)[:, 3]
predictors_df["media_sexta"]    = fc.media_por_dia(df=df_data)[:, 4]
predictors_df["media_sabado"]   = fc.media_por_dia(df=df_data)[:, 5]
predictors_df["media_domingo"]  = fc.media_por_dia(df=df_data)[:, 6]
predictors_df["vari_semana"]    = fc.vari_semanal(fc.medias(df_data), fc.media_por_dia(df=df_data))
predictors_df["dia_maximo"]     = fc.dia_maximo(df=df_data)
predictors_df["maximo"]         = fc.maximos_por_sujeito(df_data)
predictors_df["fourieramp7"]    = fourier_amp[0][:]
predictors_df["fourieramp14"]   = fourier_amp[1][:]
predictors_df["group"]          = df_groups["Group"]


print(predictors_df.head())

corr = predictors_df.corr()

print(df_groups.head())
"""df_groups = df_groups.drop(index='index', axis=0)

print(df_groups.head())"""


#_____________________Naive_Bayes_____________________#
'''
#plot = sn.heatmap(corr, annot = True, fmt=".1f", linewidths=.6)
#plt.show()
X = predictors_df.drop("group", axis=1)
y = df_groups['Group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0], (y_test != y_pred).sum()))

print(gnb.score(X_test, y_test))
print(gnb.get_params())


#plt.hist(df_groups['Group'], bins=3)
#plt.show()
'''

#_____________________MLPClassifier_____________________#

X = predictors_df.drop("group", axis=1)
y = df_groups['Group']

# Separando os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Criando um modelo MLPClassifier com 3 camadas ocultas de 10 neurônios cada
model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), learning_rate='adaptive', max_iter= 500)

# Treinando o modelo com os dados de treino
model.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliando a acurácia do modelo
y_test1 = len(y_test[y_test == 1])
y_test2 = len(y_test[y_test == 2])
y_test3 = len(y_test[y_test == 3])

acc = accuracy_score(y_test, y_pred)
print("Acurácia:", acc)
conf_mlp = confusion_matrix(y_test, y_pred)

true1, false2_was1, false3_was1, false1_was2, true2, false3_was2, false1_was3, false2_was3, true3 = conf_mlp.ravel()

print("""Matriz de confusão: 
""", conf_mlp, """
Acurácia grupo 1:""", true1/y_test1,"""
Acurácia grupo 2:""", true2/y_test2,"""
Acurácia grupo 3:""", true3/y_test3)

#________________________________________________________#

