# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator

def loadDataset(filename, trainingSet=[] ):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    next(lines)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	           dataset[x][y] = float(dataset[x][y])
	           trainingSet.append(dataset[x])
	     


def euclideanDistance(instance1, instance2, length):
	distance = 0
	#print(instance2[0].dtype)
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)
	#print(length)
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[[7.2, 3.6, 5.1, 2.5]]
	loadDataset('iris.csv', trainingSet)
	print 'Training set size: ' + repr(len(trainingSet))
	print 'Test set size: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	for k in range(5) :
	  for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k+1)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ' for k=' + repr(k+1))
	



	
main()
