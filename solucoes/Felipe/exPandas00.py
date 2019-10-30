#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:44:41 2019

@author: felipe


    1 Do arquivo carros, imprima as primeiras e as últimas cinco linhas. Faça uma versão que abre o arquivo a partir do csv e outra que abre a partir da url.
    2 Encontre o nome da empresa de carros mais cara
    3 mostre a média de preços
    4 mostre o valor mais alto e mais baixo de horsepower
    5 mostre a informação de todos os carros da marcar toyota
    6 faça uma função que filtra os carros por cilindro e salva em outro arquivo CSV


"""

import pandas as pd

from IPython.display import display

mCarros = pd.read_csv('carros.csv')

print "\nprimeiras 5 linhas"

display(mCarros.head(5))

print "ultimas 5 linhas"

display(mCarros.tail(5))

print "\na company com o veículo mais caro:"

display(mCarros[ mCarros['price']==max(mCarros.price)].iloc[0].company)

print "\nmédias dos preços:"

display( mCarros['price'].mean())

print "max | min CV :"

#display(mCarros[ mCarros['horsepower']==max(mCarros.horsepower)].iloc[0].horsepower)
#display(mCarros[ mCarros['horsepower']==min(mCarros.horsepower)].iloc[0].horsepower)

#ou fazer: 
print mCarros['horsepower'].max(), mCarros['horsepower'].min() 

print "\ncarros Toyota:"

display (  mCarros[ mCarros.company == 'toyota' ] )

def findByCilindradaToCsv(mCarros, cilindrada, path = 'cars.by.cilindrada.dat'):
    
    if cilindrada not in range(0,10): raise TypeError("Cilindrada incorreta para categoria carros")
    
    units = ['','one','two','three','four','five','six','seven','eight','nine']
    
    num = units[cilindrada]
    
    found = mCarros[ mCarros['num-of-cylinders'] == num ]
    
    found.to_csv(path)    
    
