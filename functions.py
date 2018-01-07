import math
import random
import sys
import numpy as np
from numpy import linalg as LA


"""Questa funzione prende l'oggetto myData e aggiorna:
	(1) M -> rendendola non sparsa"""
def deSparsify(myData):
	#ciclo sulle colonne (task)
	for task in range(0, myData.t):
		average = 0
		for counter in range(0, myData.p):
			average = average + myData.M[counter, task]
		average = average/myData.p
		for counter in range(0, myData.p):
			if (myData.M[counter, task] == 0):
				 myData.M[counter, task] = average

"""Questa funzione prende il file di input e l'oggetto myData e inizializza:
	(1) t -> numero di task
	(2) p -> numero di utenti
	(3) hashPeople -> lista di interi che rappresentano gli utenti
	(4) hashTasks -> lista di interi che rappresentano i task"""
def tasksAndPeopleCountAndHash(inputFile, myData):
	tasksSet = set()
	peopleSet = set()
	data = inputFile.readlines()
	lines = []
	for line in data:
		words = list(map(int, line.split()))
		tasksSet.add(words[1])
		peopleSet.add(words[0])
		lines.append(words)
		
	myData.t = len(tasksSet)
	myData.p = len(peopleSet)
	myData.hashPeople = sorted(list(peopleSet))
	myData.hashTasks = sorted(list(tasksSet))
	return lines

"""Questa funzione prende il file di input e l'oggetto myData e inizializza:
	(1) M -> matrice che ha come righe gli utenti e per colonne i task
	(2) t, p, hashPeople, hashTasks -> mediante tasksAndPeopleCountAndHash"""
def parseMatrix(inputFile, myData):
	lines = tasksAndPeopleCountAndHash(inputFile, myData)
	myData.M = np.zeros((myData.p,myData.t)) 
	for words in lines:
		myData.M[myData.hashPeople.index(words[0]), myData.hashTasks.index(words[1])] = words[2]

"""Questa funzione prende il file di input, l'oggetto myData ed un flag ed
	inizializza:
	(1) Sigmak -> matrice risultante da SVD, ma ridotta in taglia
	(2) Sk -> matrice risultante da SVD, ma ridotta in taglia
	(3) M, t, p, hashPeople, hashTasks -> mediante parseMatrix
	(4) aggiorna M (forse) -> mediante deSparsify"""
def reduceToSVDk(inputFile, myData, flag):
	#Parsa il file in una matrice
	parseMatrix(inputFile, myData)
	#Valuta il flag
	if (flag == 'yes'):
		deSparsify(myData)
	#Calcola la SVD di M
	S, Sigma, U = np.linalg.svd(myData.M, full_matrices=False)
	#Approssima M
	myData.Sigmak = np.diag(Sigma[0:myData.k])
	myData.Sk = S[...,0:myData.k]

"""Questa funzione prende l'utente user e l'oggetto myData e restituisce:
	(1) Il vettore di dimensione k, che rappresenta il concetto di quell'utente"""
def userConceptNormalized(user, myData):
	rowOfSk = myData.Sk[user]
	userCon = np.matmul(rowOfSk, myData.Sigmak)
	norm = LA.norm(userCon)
	userConNorm = userCon/norm
	return userConNorm

"""Questa funzione prende l'oggetto myData ed inizializza:
	(1) MatRes -> matrice della somiglianza tra utenti"""
def userConceptProduct(myData):
	Un = np.empty([myData.p, myData.k], dtype = float)
	for user in range(0, myData.p):
		Un[user] = userConceptNormalized(user, myData)
	myData.MatRes = np.matmul(Un, Un.transpose())
	
"""Questa funzione prende un utente useri e l'oggetto myData e ritorna:
	(1) neighboursi -> vettore delle somiglianze tra useri e gli altri utenti"""
def neighboursUi(useri, myData):
	neighboursi = np.empty([myData.p], dtype = float)
	neighboursi = myData.MatRes[useri]
	#A neigbours[useri] è assegnato un valore che non disturba la somiglianza
	neighboursi[useri] = 0
	return neighboursi

"""Questa funzione prende un utente useri e l'oggetto myData e ritorna:
	(1) indexes -> vettore degli interi che rappresentano gli h utenti più simili
		a useri, ordinati in senso decrescente"""
def closesth(useri, myData):
	indexes = np.empty([myData.h], dtype = int)
	neighboursi = neighboursUi(useri, myData)
	minimum = min(neighboursi)
	for counter in range(0, myData.h):
		indexes[counter] = np.argmax(neighboursi)
		neighboursi[np.argmax(neighboursi)] = minimum -1
	return indexes

"""Questa funzione prende l'oggetto myData, un task e indexes e restituisce:
	(1) frequency -> frequenza di un task su tutti gli h utenti di indexes"""
def taskFrequency(myData, task, indexes):
	frequency = 0
	for counter in range(0, myData.h):
		frequency = frequency + myData.M[indexes[counter], task]
	return frequency

"""Questa funzione prende un utente useri e l'oggetto myData e ritorna:
	(1) frequencies -> array delle frequenze di ogni task per quell'utente"""
def tasksFrequency(useri, myData):
	frequencies = np.empty([myData.t], dtype = int)
	indexes = closesth(useri, myData)
	for task in range(0, myData.t): 
		frequencies[task] = taskFrequency(myData, task, indexes)
	return frequencies

"""Questa funzione prende un utente useri e l'oggetto myData e restituisce:
	(1) indexesTasks -> vettore dei primi l suggerimenti (tra i task non risolti
		completamente) per l'utente useri"""
def recommendationsl(useri, myData):
	indexesTasks = np.empty([myData.l], dtype = int)
	frequencies = tasksFrequency(useri, myData)
	minimum = min(frequencies)
	counter = 0
	while counter < myData.l:
		currentTask = np.argmax(frequencies)
		frequencies[currentTask] = minimum -1
		if (myData.M[useri, currentTask] != 100):
			indexesTasks[counter] = currentTask
			counter = counter + 1
	return indexesTasks

"""Questa funzione prende la matrice suggestionsMat e l'oggetto myData e chiama
	printRow per ogni riga, NOTA: deve passare per la funzione di hash"""
def printMatrix(suggestionsMat, myData):
	for user in range(0, myData.p):
		printRow(suggestionsMat, myData.hashPeople[user], myData)

"""Questa funzione prende la matrice suggestionsMat, un utente e l'oggetto myData
	e stampa a video i suggerimenti per quell'utente applicando la funzione 
	la funzione inversa della funzione di hash sull'utente e sul task"""
def printRow(suggestionsMat, user, myData):
	print(user, end=' ')
	hashedUser = myData.hashPeople.index(int(user))
	for counter in range(0, myData.l):
		print(myData.hashTasks[suggestionsMat[hashedUser, counter]], end=' ')
	print('')		
	
