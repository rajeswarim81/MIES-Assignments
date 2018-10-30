import numpy as np

#Input array
X=np.array([[1.81,0.80,0.44],[1.77,0.70,0.43],[1.60,0.60,0.38],[1.54,0.54,0.37],[1.66,0.65,0.40],
	[1.90,0.90,0.47],[1.75,0.64,0.39],[1.77,0.70,0.40],[1.59,0.55,0.37],
	[1.71,0.75,0.42],[1.81,0.85,0.43]])

#Output
y=np.array([[0],[0],[1],[1],[0],[0],[1],[1],[1],[0],[0]])

#Test_Data
X_test=np.array([[1.63, 0.60, 0.37],[1.75, 0.72, 0.41]])

#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#ReLU Function
def ReLU(x):
    return x * (x > 0)

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)

def derivatives_ReLU(x):
	return 1 * (x>0)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.2 #Setting learning rate/ eta
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

#Forward Propogation
	hidden_layer_input1=np.dot(X,wh)
	hidden_layer_input=hidden_layer_input1 + bh
	hiddenlayer_activations = ReLU(hidden_layer_input)

	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = sigmoid(output_layer_input)

#Backpropagation
	E = y-output
	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_ReLU(hiddenlayer_activations)
	d_output = E * slope_output_layer
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	wout += hiddenlayer_activations.T.dot(d_output) *lr
	bout += np.sum(d_output, axis=0,keepdims=True) *lr
	wh += X.T.dot(d_hiddenlayer) *lr
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

#TESTING

hidden_layer_input1=np.dot(X_test,wh)
hidden_layer_input=hidden_layer_input1 + bh
hiddenlayer_activations = ReLU(hidden_layer_input)

output_layer_input1=np.dot(hiddenlayer_activations,wout)
output_layer_input= output_layer_input1+ bout
output_test = sigmoid(output_layer_input)

#Applying threshold to Output
output_pred=[]
output_pred_test=[]
for o in output:
	if(o>0.5):
		output_pred.append(1)
	if(o<=0.5):
		output_pred.append(0)
	i=i+1

for o in output_test:
	if(o>0.5):
		output_pred_test.append(1)
	if(o<=0.5):
		output_pred_test.append(0)
	i=i+1
#-------------------

#Printing Results
print('Results at the output layer when trained:')
print output
print('Classification of the training set by the Neural Network:')
print output_pred
print('Results at the ouput layer when tested:')
print output_test
print('Classification of the testing set bu the Neural Network:')
print output_pred_test
