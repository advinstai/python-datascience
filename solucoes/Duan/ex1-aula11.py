import pandas as pd
import numpy as np

# Abrir o arquivo Auto.csv 
url = 'https://raw.githubusercontent.com/advinstai/python-datascience/master/csv/Auto.csv'
df1 = pd.read_table(url,sep=",",na_values='?')  #na_values define o conteudo que deve ser interpretado como NaN
print("---")

# Contar a quantidade de elementos nulos por coluna
print(df1.isnull().sum())
print("---")

# Criar um dataframe eliminando as linhas com valores nulos dataframe 
df2 = df1.dropna()
print("Numero de linhas em df1:",df1.shape[0])
print("Numero de linhas em df2:",df2.shape[0])
print("---")

# Criar um dataframe transformando os valores nulos para zero 
df3 = df1.fillna(0)
print(df3[df3.horsepower==0])
print("---")

# Criar um dataframe trocando nulo por média da coluna 
print("Media da categoria horsepower:",round(np.mean(df2.horsepower[:]),6),"\n")    # round to 6 decimals
df4 = df1.fillna(np.mean(df2))  #usei df2 para calcular a media pois nele nao ha elementos NaN
print(df4[32:33])   # a linha 32 coluna horsepower (que era NaN) passa a conter a media 104.469388
print("---")    