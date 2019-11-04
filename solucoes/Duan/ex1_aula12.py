import pandas as pd
import numpy as np
import sqlite3


# 1 Leia os dados train.csv usando pandas
url = 'https://raw.githubusercontent.com/advinstai/python-datascience/master/Exercicios/titanic/train.csv'
df1 = pd.read_table(url,sep=",")
print("--- Dados lidos de train.csv:")
print(df1)
print("\n","-"*45)

# 2 Remova a coluna com nomes de pessoas
df2 = df1.drop('Name',axis=1)
print("--- Dados de train.csv sem a coluna 'Name':")
print(df2)
print("\n","-"*45)

# 3 A coluna Cabin contém uma letra e um número. 
#   Crie uma nova coluna chamada deck usando apenas a letra de cada registro presente na coluna Cabin.
df2['Deck'] = df2.Cabin.str[0]          # .str acessa os caracteres da string que estao naquela celula
print("--- Inserindo coluna 'Deck':")
print(df2)
print("\n","-"*45)

# 4 Substitua os valores ausentes pela média de valores da coluna
df3 = df2.dropna()
df4 = df2.fillna(round(np.mean(df3)))      # pega valores NaN em df2 e preenche com media de df3, que nao tem NaN
print("--- Substituindo valores ausentes por media:")
print(df4)
print("\n","-"*45)

# 5 Os valores da coluna "embarked" sao os seguintes: C = Cherbourg, Q = Queenstown, S = Southampton
# 6 Crie um CSV com essas uma coluna nome_cidade_de_embarque: 
#   C = Cherbourg, Q = Queenstown, S = Southampton e mais uma coluna chamada população com valores hipotéticos, 

print("\n--- Adicionando coluna populacao e salvando arquivo CSV:\n")
new_df = pd.DataFrame([['C',42318],['Q',15650],['S',228600]],columns=['nome_cidade_de_embarque','populacao'])
new_df.to_csv('nome_cidade_embarque.csv',index=False)
print(new_df)
print("\n","-"*45)

#   faça um merge entre o CSV train.csv e esse novo CSV com os nomes das cidades. 
print("\n--- Fazendo MERGE entre train.csv e nome_cidade_embarque.csv:\n")
mergedDf = pd.merge(df4,new_df,how='left', left_on="Embarked",right_on="nome_cidade_de_embarque")
print(mergedDf)
print("\n","-"*45)

#   Implemente um filtro de passageiros por população da cidade de embarque
print("\n--- Filtro de passageiros por populacao da cidade de embarque:\n")
df_com_nome = pd.merge(df1,new_df,how='left',left_on="Embarked",right_on="nome_cidade_de_embarque")
def filtroPop(pop):
    #return df_com_nome[df_com_nome.populacao == pop]
    return mergedDf[mergedDf.populacao == pop]

#print(filtroPop(42318))         # 168 passageiros neste filtro
#print(filtroPop(15650))         #  77 passageiros neste filtro
print(filtroPop(228600))        # 644 passageiros neste filtro