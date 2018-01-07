#!/usr/bin/env python3
import math
import random
import sys
import numpy as np
import readline
from numpy import linalg as LA
from functions import *
"""Struttura dati necessaria per lo storage compattato delle informazioni"""


class Useful:
    def __init__(self, k, h, l):
        self.k = k
        self.h = h
        self.l = l


k = 28  #Miglior valore dai test
h = 61  #Numero di vicini da considerare, miglior valore dai test
l = 20  #Numero di suggerimenti, a scelta

print('Recommendation system for CMS')
inputFile = open("punteggi.txt", "r")
var = input(
    'Press y for recommendations from desparsified matrix, press n for recommendations from the original one -> '
)
myData = Useful(k, h, l)
if (var == 'y'):
    reduceToSVDk(inputFile, myData, 'yes')
else:
    reduceToSVDk(inputFile, myData, 'no')
suggestionsMatrix = np.empty([myData.p, myData.l], dtype=int)
userConceptProduct(myData)
for useri in range(0, myData.p):
    indexesTasks = recommendationsl(useri, myData)
    for task in range(0, myData.l):
        suggestionsMatrix[useri, task] = indexesTasks[task]
#Comunicazione risultati
var = input(
    'Press a for all the results, press u for a specific user s suggestions -> '
)
#Nota: solo nell'operazione di print sono applicate le funzioni inverse delle
#funzioni di hash
if (var == 'a'):
    printMatrix(suggestionsMatrix, myData)
else:
    user = input('Insert the user number -> ')
    printRow(suggestionsMatrix, user, myData)
