import csv
import os
import numpy as np

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter = {'float_kind':float_formatter})

#sigmoid acrtivation function and derivate of sigmoid function
def sigmoid(x, deriv = False):
	#derivative of sigmoid is  x/1-x
	if(deriv == True):
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

#linear activation function for output layer - not working well 
def linear(x, deriv = False):
	#derivative of y=x is 0
	if (deriv == True):
		return 0
	return x

#get data from csv and plop into arrays
def getData(file):
	global outputs;
	global inputs;
	#load all data
	os.chdir("D:\Scripts\AI_example")
	data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)
	#get shape of array [x:y]
	shape = data.shape
	numRows = shape[0]
	numCols = shape[1]

	outputs = data[:,numCols-1].reshape(numTraining, 1)
	inputs = np.delete(data, numCols-1, axis=1)


#training data
numInputs = 3
numOutputs = 1
numTraining = 40

getData("data.csv")


#renew seed - internet says this is good
np.random.seed(1)

#randomly generate inital weights from -1 to 1
synapse0 = 2*np.random.random((numInputs, numTraining)) - 1
synapse1 = 2*np.random.random((numTraining, numOutputs)) - 1

for i in range(50000):
	# Feed forward through layers 0, 1, and 2
	layer0 = inputs;
	layer1 = sigmoid(np.dot(layer0, synapse0))
	layer2 = sigmoid(np.dot(layer1, synapse1))
	#calc error of output
	layer2Err = outputs - layer2

	#periodically print error
	if (i% 5000) == 0:
		print ("Error for " + str(i) + " iteration: " + str(np.mean(np.abs(layer2Err))))
		print(layer2)

	#calc derivate of output, see if it is heading in the right direction
	layer2Delta = layer2Err * sigmoid(layer2, deriv = True)
	#find contribution of layer 1 inputs to layer 2 (output)
	layer1Err = layer2Delta.dot(synapse1.T)
	#calc derivate of layer 1, see if it is heading in the right direction
	layer1Delta = layer1Err * sigmoid(layer1 ,deriv = True)

	#change weights, then run again
	synapse1 += layer1.T.dot(layer2Delta)
	synapse0 += layer0.T.dot(layer1Delta)

print("Network trained")
print("")


#loop through to get new inputs - does not continue to train yet
while True:
	newInput = [0,0,0]
	newInput[0] = int(input("enter new input1: "))
	newInput[1] = int(input("enter new input2: "))
	newInput[2] = int(input("enter new input3: "))

	print(newInput)

	layer0 = newInput;
	layer1 = sigmoid(np.dot(layer0, synapse0))
	layer2 = sigmoid(np.dot(layer1, synapse1))
	print(layer2)
