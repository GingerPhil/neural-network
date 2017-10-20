import csv
import os
import numpy as np

#set number digits after decimal point
float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter = {'float_kind':float_formatter})

#sigmoid activation function and derivate of sigmoid function
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
	global inputs;
	global scaledOutputs;
	global scaleFactor;
	#load all data
	os.chdir("D:\Scripts\AI_example")
	data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1)
	#get shape of array [x:y]
	shape = data.shape
	numRows = shape[0]
	numCols = shape[1]

	outputs = data[:,numCols-1].reshape(numTraining, 1)
	scaleFactor = np.amax(outputs)
	scaledOutputs = outputs/scaleFactor

	inputs = np.delete(data, numCols-1, axis=1)

alphas = [0.001]
hiddenSize = 40

#training data
numInputs = 3
numOutputs = 1
numTraining = 40

getData("data.csv")


for alpha in alphas:
	print ("\nTraining With Alpha:" + str(alpha))
	#renew seed - internet says this is good
	np.random.seed(1)
 
	# randomly initialize our weights with mean 0
	synapse_0 = 2*np.random.random((numInputs, hiddenSize)) - 1
	synapse_1 = 2*np.random.random((hiddenSize, numOutputs)) - 1
 
	for j in range(10000000):
 
		# Feed forward through layers 0, 1, and 2
		layer_0 = inputs
		layer_1 = sigmoid(np.dot(layer_0, synapse_0))
		layer_2 = sigmoid(np.dot(layer_1, synapse_1))
	 
		# how much did we miss the target value?
		layer_2_error = layer_2 - scaledOutputs
 
		if (j% 1000000) == 0:
			print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

		# in what direction is the target value?
		# were we really sure? if so, don't change too much.
		layer_2_delta = layer_2_error * sigmoid(layer_2, deriv=True)
 
		# how much did each l1 value contribute to the l2 error (according to the weights)?
		layer_1_error = layer_2_delta.dot(synapse_1.T)
 
		# in what direction is the target l1?
		# were we really sure? if so, don't change too much.
		layer_1_delta = layer_1_error * sigmoid(layer_1, deriv=True)
 
		synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
		synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))


#loop through to get new inputs - does not continue to train yet
print(scaledOutputs)
print(layer_2 * scaleFactor)
