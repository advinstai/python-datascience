import numpy as np
import pandas as pd

# Create an array of ones
n = np.ones((3,4))
print("np.ones: ",n)
# Create an array of zeros
n=np.zeros((2,3,4),dtype=np.int16)
print("np.zeros: ",n)
# Create an array with random values
n=np.random.random((2,2))
print("random.random: ",n)
# Create an empty array
n=np.empty((3,2))
print("empty: ",n)
# Create a full array
n=np.full((2,2),7)
print("full: ",n)
# Create an array of evenly-spaced values
n=np.arange(10,25,5)
print("arange: ",n)
# Create an array of evenly-spaced values
n=np.linspace(0,2,9)
print("linspace: ",n)

n = np.eye(5)
print("np.eye: ",n)

n = np.identity(5)
print("np.identity: ",n)

# Import your data
x, y, z = np.loadtxt('text-file.txt', skiprows=1, unpack=True)

print("loadtxt ")
print("x: ",x)
print("y: ",y)
print("z: ",z)

my_array2 = np.genfromtxt('text-file.txt',
                      skip_header=1,
                      filling_values=-999)

print("my_array2: ",my_array2)

n = np.eye(5)
m=np.full((2,2),7)

np.save('test',n)

nHIGH=np.eye(50)
np.savez('testH',nHIGH)
np.savez_compressed('testH-c',nHIGH)

my_array=np.full((12,12),70)
# Print the number of `my_array`'s dimensions
print(my_array.ndim)

# Print the number of `my_array`'s elements
print(my_array.size)

# Print information about `my_array`'s memory layout
print(my_array.flags)

# Print the length of one array element in bytes
print(my_array.itemsize)

# Print the total consumed bytes by `my_array`'s elements
print(my_array.nbytes)
np.savez('my_array',my_array)

x=np.random.random((3,4))
y=np.random.random((5,1,4))

# Add `x` and `y`
res=np.add(x,y)
print('res: ',res)

res2=res.sum()
print('res2: ',res2)

my_2d_array=np.random.random((4,4))
# Select the element at row 1 column 2
print(my_2d_array[1][2])
# Select the element at row 1 column 2
print(my_2d_array[1,2])

#a[start:end] # items start through the end (but the end is not included!)
#a[start:]    # items start through the rest of the array
#a[:end]      # items from the beginning through the end (but the end is not included!)

data = np.array([['','Col1','Col2','Col3'],
                ['Row1',1,2,6],
                ['Row2',3,4,7]])

print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))



# Take a 2D array as input to your DataFrame
my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
print(pd.DataFrame(my_2darray))

print("====")

# Take a dictionary as input to your DataFrame
my_dict = {1: ['1', '3'], 2: ['1', '2'], 3: ['2', '4']}
print(pd.DataFrame(my_dict))

print("====")

# Take a DataFrame as input to your DataFrame
my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['A'])
print(pd.DataFrame(my_df))

# Take a Series as input to your DataFrame
my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington", "Belgium":"Brussels"})
print(pd.DataFrame(my_series))

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))

# Use the `shape` property
print(df.shape)

# Or use the `len()` function with the `index` property
print(len(df.index))

dataf = np.array([['','A', 'B', 'C'], [0,1, 2, 3], [1,4, 5, 6], [2,7, 8, 9]])
print(dataf)
print(dataf[0,1:])

df=pd.DataFrame(data=dataf[1:,1:],
                  index=dataf[1:,0],
                  columns=dataf[0,1:])

print(df)


# Using `iloc[]`
print(df.iloc[0][0])

# Using `loc[]`
#print(df.loc[0]['A'])

# Using `at[]`
#print(df.at[0,'A'])

# Using `iat[]`
print(df.iat[0,0])


print("=======================")
#df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= np.random.random((3)), columns=[48, 49, 50])

df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= np.zeros((3)), columns=[48, 49, 50])


print(df)
df_reset = df.reset_index(level=0, drop=True)
print(df_reset)
print("=======================")
print("=======================")

# Pass `2` to `loc`
#print(df.loc[2])

# Pass `2` to `iloc`
#print(df.iloc[2])

# Pass `2` to `ix`
#print(df.ix[2])
#print("=======================")
#df_reset = df.reset_index(level=0, drop=True)
#print("=======================")
#print(df_reset)
#print("=======================")
#df_reset2 = df.reset_index(level=0, inplace=True)
#print("=======================")
#print(df_reset2)
#print("=======================")
#print(df)
#print("=======================")

#n=np.zeros((2,3,4),dtype=np.int16)

# Check out the DataFrame `df`
#print(df)

# Drop the column with label 'A'
#df.drop('A', axis=0, inplace=True)

# Drop the column at position 1
ddrop=df.drop(df.columns[[-1]], axis=1)

#df[1,1]=100
#print(df)
print(ddrop)

df.rename(index={2: 'a'})
print(df)



df = pd.DataFrame(index=range(0,4),columns=['A'], dtype='float')
print(df)





















* Create a null vector of size 10
* How to find the memory size of any array
* Create a null vector of size 10 but the fifth value which is 1
* Create a vector with values ranging from 10 to 49
* Reverse a vector (first element becomes last)
* Create a 3x3 matrix with values ranging from 0 to 8
* Find indices of non-zero elements from \[1,2,0,0,4,0\]
* Create a 3x3 identity matrix
* Create a 3x3x3 array with random values
* Create a 10x10 array with random values and find the minimum and maximum values
* Create a random vector of size 30 and find the mean value
* Create a 2d array with 1 on the border and 0 inside
* Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
* Create a 8x8 matrix and fill it with a checkerboard pattern
* Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
* Create a structured array representing a position (x,y) and a color (r,g,b
* How to find the most frequent value in an array?



* Crie um vetor nulo de tamanho 10
* Como encontrar o tamanho da memória de qualquer matriz
* Crie um vetor nulo de tamanho 10, mas o quinto valor, que é 1
* Crie um vetor com valores que variam de 10 a 49
* Inverter um vetor (o primeiro elemento se torna o último)
* Crie uma matriz 3x3 com valores que variam de 0 a 8
* Encontre índices de elementos diferentes de zero em \ [1,2,0,0,4,0 \]
* Crie uma matriz de identidade 3x3
* Crie uma matriz 3x3x3 com valores aleatórios
* Crie uma matriz 10x10 com valores aleatórios e encontre os valores mínimo e máximo
* Crie um vetor aleatório de tamanho 30 e encontre o valor médio
* Crie uma matriz 2D com 1 na borda e 0 dentro
* Crie uma matriz 5x5 com valores 1,2,3,4 logo abaixo da diagonal
* Crie uma matriz estruturada representando uma posição (x, y) e uma cor (r, g, b)
* Subtrair a média de cada linha de uma matriz
* Como encontrar o valor mais frequente em uma matriz?
* Crie uma matriz a partir de um arquivo
* crie uma matriz com valores aletaórios e salve para um arquivo








#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


```python
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)


```python
# Author: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```

#### 69. How to get the diagonal of a dot product? (★★★)


```python
# Author: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version  
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)
```

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)


```python
# Author: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```

#### 72. How to swap two rows of an array? (★★★)


```python
# Author: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python
# Author: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python
# Author: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```

#### 75. How to compute averages using a sliding window over an array? (★★★)


```python
# Author: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)


```python
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python
# Author: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)


```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)


```python
# Author: Italmassov Kuanysh

# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python
# Author: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)


```python
# Author: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```

#### 82. Compute a matrix rank (★★★)


```python
# Author: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```

#### 83. How to find the most frequent value in an array?


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python
# Author: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```

#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)


```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python
# Author: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


```python
# Author: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```

#### 88. How to implement the Game of Life using numpy arrays? (★★★)


```python
# Author: Nicolas Rougier

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```

#### 89. How to get the n largest values of an array (★★★)


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])
```

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


```python
# Author: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

#### 91. How to create a record array from a regular array? (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python
# Author: Ryan G.

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python
# Author: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)


```python
# Author: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```

#### 95. Convert a vector of ints into a matrix binary representation (★★★)


```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python
# Author: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# Author: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
# Author: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


```python
# Author: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)


```python
# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```

