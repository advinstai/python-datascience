import pandas as pd

# 1 Do arquivo carros, imprima as primeiras e as últimas cinco linhas. Faça uma versão que abre o arquivo a partir do csv e outra que abre a partir da url.

# open file
def open_csv_file(file_path):
    df = pd.read_csv(file_path)
    print(df.iloc[0:5,])
    print(df.tail(5))

print(open_csv_file('/home/vitorbezerra/Documents/python-datascience/Exercicios/carros.csv'))

# open url

def open_csv_url(url):
    df = pd.read_table(url, sep=',')
    print(df.iloc[0:5,])
    print(df.tail(5))

print(open_csv_url('https://raw.githubusercontent.com/vitorbezerra/python-datascience/master/Exercicios/carros.csv'))

# 2 Encontre o nome da empresa de carros mais cara

df = pd.read_table('https://raw.githubusercontent.com/vitorbezerra/python-datascience/master/Exercicios/carros.csv', sep=',')
print(df.loc[df['price'] == df['price'].max() ,])

# 3 mostre a média de preços
print(df['price'].mean())

# 4 mostre o valor mais alto e mais baixo de horsepower
print("Horsepower max: {} min:{}".format(df['horsepower'].max(),df['horsepower'].min()))

# 5 mostre a informação de todos os carros da marcar toyota
print('Toyota cars:')
print(df.loc[df['company'] == 'toyota',])

# 6 faça uma função que filtra os carros por cilindro e salva em outro arquivo CSV
def filter_cylinders(df, number_cylinders):
    df_cylinders = df.loc[df['num-of-cylinders'] == number_cylinders, ]
    df_cylinders.to_csv('cylinders.csv')

filter_cylinders(df,'six')