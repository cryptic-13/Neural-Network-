import numpy as np 
import matplotlib.pyplot as plt 

#as there is no input from back propagation for the first trial, the parameters must be randomly initialized 
# parameters --> weight matrix and bias vector 

def initial_parameters(layer_dims) :  
    np.random.seed(3) 
    paramters= {} 
    numLayers= len(layer_dims) 
     #numLayersL= number of layers in the network, layer_dims is a list of dimensions of each layer
    for x in range 1 to numLayers:
        parameters['W'+str(x)]= np.random.randn(ayer_dims(1), layer_dims(x-1))*0.01 
        parameters['b'+str(x)]= np.zeros((layer_dims(x),1))
    return parameters 


    #define sigmoid function, which takes the input from previous layer
    #  --> ΣW(i)A(i) +B (for i ranges from 1 to total number of neurons in the previous layer) 
    # W(i) is the corresponding element (matched by 'i' value) in the weight matrix and 
    # A(i) is the activation level of the ith neuron in the previous layer 
    # to find the activation level of a neuron in the current layer, sigmoid takes this summation and squishes it to a value bw 0 and 1 


def sigmoid(z): 
        A= 1/(1 +(e**(-z)))
        cache= z
        return (A,cache) 

    #cache is returned to facilitate future backpropagation 
    #In forward propagation, the first layer takes its input from the database, processes its output using parameters and input 
    #output is passed to next layer and so on 


def forward_prop(input, parameters) : 
        A= input 
        #assign the input to activation level of first layer of neurons 
        caches= []
        neuronCount= len(parameters)//2 
        #because for each neuron, 2 parameters exist, weight and bias 
        for x in range (1, neuronCount+1) : 
            A_prev= A
            #store the initial input to first layer neuron 
            z= np.dot( parameters['W'+str(x)] ,A_prev ) + parameters['b'+str(x)] 
            # z represents the activation of each neuron in the first layer, after squishification using sigmoid fn
            # this is the output of the first layer, using the linear hypothesis formula 

            linear_cache= (A_prev, parameters['W', str(x)], parameters['b',str(x)])

            A, activ_cache= sigmoid(z)

            cache= (linear_cache, active_cache)
            caches.append(cache)

            #linear_cache contains all the individual components that calculated the expression to which sigmoid was appplied 
            #linear_cache stores the activation levels, weights and biases as 3 separate entities 
            #activation cache store the linear hypothesis post matrix multiplication -->  ΣW(i)A(i) +B --> this value 


def cost_fn(predicted, truth) :
                A= predicted 
                #a matrix of all the final output (predictions) made by the neural network 
                #truth is a vector/matrix of all the actual values or true values taken from the training dataset 
                #size= number of data points in training dataset 

                size= truth.shape[1]
                cost= (-1/size) * (np.dot(np.log(predicted), truth.T) + np.dot (log(1-predicted),1-truth.T))
                return cost 

            
def chain_rule_logic(dA,cache): 
                linear_cache, activ_cache= cache 
                z= active_cache 
                dz=  dA*sigmoid(z) * (1-sigmoid(z)) 
                #derivative of sigmoid function wrt lienar output of that neuron 

                A_prev, W,b= linear_cache 
                count= A_prev.shape[1]
                #count = number of neurons in previous layer 
                dW= (1/count) * (np.dot(dZ, A_prev.T))
                db= (1/count) * np.sum(dZ, axis=1, keepdims=True)
                dA_prev= np.dot (W.T, dZ)

                #dw, db, dA are the derivatives of the cost function wrt weights, biases and previous activations. 

                return (dA_prev, dW, db)


def backprop (A_last, truth, caches) : 
    #A_last is a vector of activations of the last layer of neurons, ergo the networks final predictions 
    # truth is the vector of actual outputs, or true labels 
    
    layerCount= len (caches) 
    final_neuron_count= A_last.shape[1]
    truth.reshape(A_last.shape)

    dAL= -(np.divide(truth, A_last) - np.divide(1-truth, 1-A_last))
    current_cache=  caches[layerCount-1]
    L= layerCount

    #create a dictionary of all the gradients calculated for backpropagation 
    # each gradient is a partial derivative of the cost fn wrt each parameter
    # each layer will have a different set of A1, W1, b1 --> parameters and activation
    #this you can get from its 'caches' 
    # so, for each layer, 3 gradients are computed and stored as a tuple. 
    # this process is repeated for as many layers as there are 

    gradients ['dA'+str(L-1)], gradients ['dW'+ str(L-1)], gradients ['db'+str(L-1)] = chain_rule_logic(dAL, current_cache)

    for l in reversed(range(L-1)): 
        current_cache= caches[l]
        dA_prev_temp, dW_temp, db_temp= chain_rule_logic(gradients['dA'+str(l+1)], current_cache)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

return (grads) 


def update_parameters (parameters, gradients, learning_rate) : 
    
      L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] -learning_rate*grads['W'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] -  learning_rate*grads['b'+str(l+1)]

    return parameters


def train(X, Y, layer_dims, epochs, lr):
    params = initialize_parameters(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, parameters)
        cost = cost_fn(Y_hat, Y)
        cost_history.append(cost)
        gradients = backprop(Y_hat, Y, caches)

        parameters = update_parameters(parameters, gradients, lr)


    return parameters, cost_history

            



