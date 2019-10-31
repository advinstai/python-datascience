# python-datascience

## Pandas

```
import pandas as pd
import numpy as np
```

## Series
```
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'])
s
```

## Mudando Índices
```
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],
              index=['A', 'Z', 'C', 'Y', 'E'])
s
```

## Series a partir de dicionário
```
d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,
     'Austin': 450, 'Boston': None}
cities = pd.Series(d)
cities
```

## Selecionando dados
```
cities[cities < 1000]
```

## Mostrando itens da series que atendem a condição
```
less_than_1000 = cities < 1000
print(less_than_1000)
print('\n')
print(cities[less_than_1000])
```

## Cálculos com series
```
cities/3
np.square(cities)
```

## trabalhando com nulos
```
print(cities)
cities.notnull()

print(cities.isnull())
print('\n')
print(cities[cities.isnull()])
```

## Carregando um DataFrame a partir de um CSV
```
from_csv = pd.read_csv('teste.csv')
from_csv.head()
```

## Carregando um DataFrame a partir de uma URL
```
url = 'https://raw.github.com/gjreda/best-sandwiches/master/data/best-sandwiches-geocode.tsv'

# fetch the text from the URL and read it into a DataFrame
from_url = pd.read_table(url, sep='\t')
from_url.head()
```

## Exemplo com loc, iloc e ix
```
df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2, 'A', 4], columns=[48, 49, 50])

# Pass `2` to `loc`
print(df.loc[ 'A'])

# Pass `2` to `iloc`
print(df.iloc[1])

# Pass `2` to `ix`
print(df.ix[2])

df3 = from_url.iloc[:,[0,2]]
print(df3)
print(df3.loc[ df3['rank']>10])

```

## conversao para int
```
df["normalized-losses"]=df["normalized-losses"].astype(int)
```
