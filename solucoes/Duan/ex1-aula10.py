import pandas as pd
import numpy as np

#1 Do arquivo carros, imprima as primeiras e as últimas cinco linhas. 
#  Faça uma versão que abre o arquivo a partir do csv e outra que abre a partir da url.

print("--- Exercicio 1:")

url="https://raw.githubusercontent.com/advinstai/python-datascience/master/Exercicios/carros.csv"
fromurl = pd.read_table(url,sep=",")
	#fromcsv = pd.read_csv()
print("carros.csv - 5 primeiras linhas:")
print(fromurl.head())
print("")
print("carros.csv - 5 ultimas linhas:")
print(fromurl.tail())
print("---")

#2 Encontre o nome da empresa de carros mais cara

print("--- Exercicio 2:")

max = pd.Series.max(fromurl['price'])
print('Nome da empresa de carros mais cara:')
print(fromurl[fromurl.price==max].company)
print("--- NAO SEI PQ A LINHA ACIMA FOI IMPRESSA ---")
print("---")

#3 mostre a média de preços

print("--- Exercicio 3:")

print("Media de precos:",round(fromurl.price.describe().mean(),2))
print("---")

#4 mostre o valor mais alto e mais baixo de horsepower

print("--- Exercicio 4:")

maxHP = pd.Series.max(fromurl.horsepower)
minHP = pd.Series.min(fromurl.horsepower)
print("HP mais alto:", maxHP)
print("HP mais baixo:", minHP)

print("---")

#5 mostre a informação de todos os carros da marcar toyota

print("--- Exercicio 5:")

print(fromurl[fromurl.company == "toyota"])
print("---")

#6 faça uma função que filtra os carros por cilindro e salva em outro arquivo CSV

print("--- Exercicio 6:")

def filterCylinder(df,strNum):
    filteredDF = df[df["num-of-cylinders"]==strNum]
    filteredDF.to_csv("filteredDF.csv")
    print("Arquivo .csv gerado com sucesso!")
    return True

filterCylinder(fromurl,"four")
print("---")

# EXTRA: Usar funcao .describe() em arquivo .csv:

print("--- Exercicio EXTRA .describe():")

url = "https://raw.githubusercontent.com/advinstai/python-datascience/master/csv/gpa.csv"
df = pd.read_table(url,sep=" ")
print(df.describe())
print("---")
