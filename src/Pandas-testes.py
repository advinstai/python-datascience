import numpy as np

np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

#big_array = np.random.randint(1, 100, size=1000000)
big_array = np.random.randint(1, 100, size=1000)
get_ipython().run_line_magic('timeit', 'compute_reciprocals(big_array)')

print(compute_reciprocals(values))
print(1.0 / values)


import pandas as pd



area = pd.Series({'California': 423967, 'Texas': 695662,
'New York': 141297, 'Florida': 170312,
'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127, 'Florida': 19552860,
'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data


data.area




data['density'] = data['pop'] / data['area']
data['density']




data['density'] = data.pop / data.area



print(type(data.values))



data['Florida':'Illinois']



data[data.density > 100]




df = pd.DataFrame([0, 10, 3, 4],
columns=['A', 'B', 'C', 'D'])
df




A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])

A + B

A.add(B, fill_value=0)




import random  as rng

rng.randint(0, 10)




A = pd.DataFrame(np.random.randint(0, 20, (2, 2)),
columns=list('AB'))
A




A = np.random.randint(10, size=(3, 4))
df - df.iloc[0]
print(A)
A - A[0]
df = pd.DataFrame(A, columns=list('QRST'))
print(df - df.iloc[0])

df.subtract(df['R'], axis=0)




halfrow = df.iloc[0, ::2]
halfrow
df - halfrow




for dtype in ['object', 'int']:
    print("dtype =", dtype)
    get_ipython().run_line_magic('timeit', 'np.arange(1E6, dtype=dtype).sum()')
    print()




vals1 = np.array([1, None, 3, 4])
vals1.sum()
vals2 = np.array([1, np.nan, 3, 4])
vals2.sum()




z=pd.Series([1, np.nan, 2, None, "teste"])
z




data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())
print(data.notnull())
d22=data.dropna()
data.dropna()
print(d22)
print(data)




print(data)
d22=data.fillna(0)
print(data)
print(d22)




index = [('California', 2000), ('California', 2010),
('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
18976457, 19378102,
20851820, 25145561]
pop = pd.Series(populations, index=index)
pop




pop[('California', 2010):('Texas', 2000)]




pop[[i for i in pop.index if i[1] == 2010]]





print(index)
index = pd.MultiIndex.from_tuples(index)
print(index)




pop = pop.reindex(index)
print(pop)
pop[:, 2010]




def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)


df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1); print(df2); print(pd.concat([df1, df2]))

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
#print(df3); print(df4); print(pd.concat([df3, df4], axis=1))
print(df3); print(df4); print(pd.concat([df3, df4]))

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])

y.index = x.index # make duplicate indices!
print("\n\n")
print(x); print(y); print(pd.concat([x, y]))

#print(x); print(y); print(pd.concat([x, y], verify_integrity=True))
print(x); print(y); print(pd.concat([x, y], ignore_index=True))



df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue','h'],
'group': ['Accounting', 'Engineering', 'Engineering', 'HR','gg']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue','g'],
'hire_date': [2004, 2008, 2012, 2014,2016]})
print(df1); print(df2)
df3 = pd.merge(df1, df2)
df3




df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4))




df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
'Engineering', 'Engineering', 'HR', 'HR'],'skills': ['math', 'spreadsheets', 'coding', 'linux',
'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))

#conversao para int
df["normalized-losses"]=df["normalized-losses"].astype(int)

#sort
df.sort_values(by=['Brand'], inplace=True)
               
