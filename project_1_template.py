# This is the template code for Programming Project 1 - 
# Option 1: Backpropagation and Autoencoders.
#
# You can add command-line arguments if you'd like (see below __main__).
#
# You can add 'parameters' to the functions in the respective arguments.
# They are called 'initialization_params', 'training_params', and 'test_params'.
# Think of these as means of controlling how your code works under different
# settings. They are supposed to be python lists, so you can use them eg.:
#    learning_rate = training_params[0]
#    stopping_criterion = training_params[1]
# etc.
# You may also not use them if you do not need them, eg. if initialization
# is completely random, initialization_params may not be necessary.
#
# XXX_state (eg. "feedforward_classifier_state") is a variable that is supposed to hold
# current firing status of all the nodes in the network.
# XXX_connections (eg. "feedforward_classifier_connections") is a variable that is supposed
# to hold the weights of the connections between the nodes.
# You have flexibility designing these variables.
# eg, you may have a flat vector of dimensionality N for XXX_states, where N is the total number
# of the nodes in the network, or you can as well have a list of state vectors, where each 
# element (vector) in the list corresponds to one of the layers.
# You can as well design XXX_connections variables as you wish, ie, use a flattened or 
# layer-by-layer representation, or anything you would like really.
#
# You do not necessarily have to use every argument in every function, 
# ie. feedforward_classifier_state argument is probably redundant in test_feedforward_classifier().
#
# You are very welcome to add more functions: They can be auxilary 
# functions, plotting functions, anything you wish.
#
# Also feel free to add global variables if you like.
#
# In __main__ as well, you can add any additional code anywhere.
# eg., you may consider having a loop for varying parameters.
# You may also change the "flow" of the code, eg, you can collect
# test data after training is complete: You have flexibility with
# this code part really.
#
# For your "matrices":
# You can use either the array or matrix type of numpy, or even switch 
# between them as you wish.
#
# Enjoy! :)
# Hande
 
 
import numpy as np
import mnist_load_show as mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,stats
# Uncomment if you will use the GUI code:
# import gui_template as gt


#Parameters

# Allow switching nonlinearities
def tanh_derivative(x):
    return np.square(np.tanh(x)) + (-1)
nonlinear = np.tanh
nonlinear_derivative = tanh_derivative

train_size = 5000
test_size = 5000
learning_rate = 0.5
training_runs = 5000

# Initialize the corresponding networks
def init_feedforward_classifier(initialization_params):
    # Extract parameters
    layer_sizes = initialization_params[0]
    num_layers = layer_sizes.shape[0]
    max_layer_size = layer_sizes.max()

    # Initially, no neurons are firing, so state is all zeros
    # Except for thresholds, which we'll put 0.1 in for now.
    neuron_states = np.zeros((num_layers, max_layer_size))
    threshold_values = np.ones(num_layers - 1)
    feedforward_classifier_state = [neuron_states, threshold_values]

    # Initialize classifier weights.  Use a 3D array, layers x inputs x outputs.
    # Threshold weights too, but only a 2D needed for those
    # Also set weights for neurons which do not exist to 0
    # Since those weights will never contribute to sums in subsequent layers,
    # backpropagation should never cause them to change
    layer_weights = np.random.random((num_layers - 1, max_layer_size, max_layer_size))
    threshold_weights = np.random.random((num_layers - 1, max_layer_size))
    for l in np.arange(num_layers - 1):
        layer_weights[l][layer_sizes[l]:max_layer_size] = 0
        layer_weights[l][:][layer_sizes[l+1]:max_layer_size] = 0
        threshold_weights[l][layer_sizes[l]:max_layer_size] = 0
    feedforward_classifier_connections = [layer_weights, threshold_weights]

    return [feedforward_classifier_state, feedforward_classifier_connections]

def init_autoencoder(initialization_params):
    # Place your code here
    return [autoencoder_state, autoencoder_connections]

def init_autoencoder_classifier(initialization_params):
    # Place your code here
    return [autoencoder_classifier_state, autoencoder_classifier_connections]


# Given an input, these functions calculate the corresponding output to 
# that input.
def update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections):
    neuron_states = feedforward_classifier_state[0]
    threshold_values = feedforward_classifier_state[1]
    layer_weights = feedforward_classifier_connections[0]
    threshold_weights = feedforward_classifier_connections[1]

    for l in np.arange(1, feedforward_classifier_state.shape[0]):
        input_sum = layer_weights[l-1].dot(neuron_states[l-1])
        input_sum += threshold_values[l-1] * threshold_weights[l-1]
        input_sum = nonlinear(input_sum)

    return feedforward_classifier_state
    
def update_autoencoder(autoencoder_state, autoencoder_connections):
    # Place your code here
    return autoencoder_state
    
def update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections):
    # Place your code here
    return autoencoder_classifier_state
    
        
        
# Main functions to handle the training of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the update functions.
def train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params):
    num_runs = training_params[0]
    data = training_data[0]
    labels = training_data[1]
    neuron_states = feedforward_classifier_state[0]

    # Stochastic gradient descent
    for i in np.arange(num_runs):
        rand_index = np.random.randint(0, data.shape[0])
        datum = data[rand_index]
        label = labels[rand_index]
        neuron_states[0] = datum
        update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        feedforward_classifier_connections = backpropagate_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, label)

    output_feedforward_classifier_performance(feedforward_classifier_state, feedforward_classifier_connections, training_data)
    return feedforward_classifier_connections
    
def train_autoencoder(autoencoder_state, autoencoder_connections, training_data, training_params):
    # Place your code here
    
    # Please do output your training performance here
    return autoencoder_connections

def train_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params):
    # Place your code here
    
    # Please do output your training performance here
    return autoencoder_classifier_connections


# Backpropagation functions
def backpropagate_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, label):
    label_vector = np.zeros(10)
    label_vector[label] = 1
    neuron_states = feedforward_classifier_state[0]
    num_layers = neuron_states.shape[0]
    max_layer_size = neuron_states.shape[1]
    layer_weights = feedforward_classifier_connections[0]
    threshold_weights = feedforward_classifier_connections[1]

    weight_changes = np.npdarray((num_layers - 1, max_layer_size))
    threshold_changes = np.ndarray(num_layers - 1)

    output_delta = (neuron_states[:-1] - label_vector) * nonlinear_derivative(neuron_states[:-1])
    prev_delta = output_delta
    for l in np.arange(1, num_layers-1)[::-1]:
        weight_delta_sums = layer_weights[l].dot(prev_delta)
        curr_delta = nonlinear_derivative(neuron_states[l]) * weight_delta_sums
        threshold_delta = np.sum(threshold_weights[l] * prev_delta)
        weight_changes[l] = -learning_rate * curr_delta * neuron_states[l-1]
        threshold_changes[l] = -learning_rate * threshold_delta
        prev_delta = curr_delta

    layer_weights += weight_changes
    threshold_weights += threshold_changes
    return feedforward_classifier_connections

# Functions for outputing the results of an ANN on a data set
def output_feedforward_classifier_performance(feedforward_classifier_state, feedforward_classifier_connections, check_data)
    data = check_data[0]
    labels = check_data[1]
    neuron_states = feedforward_classifier_state[0]

    correct = 0
    for i in np.arange(data.shape[0]):
        neuron_states[0] = data[i]
        update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        prediction = np.argmax(neuron_states[:-1])
        correct += 1 if prediction == labels[i] else 0

    print (correct / data.shape[0])
    None

# Main functions to handle the testing of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the 'run' functions.
def test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None
    
def test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None

def test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params):
    # Place your code here
    
    # Please do output your test performance here
    None
    


# You may also want to be able to save and load your networks, which will 
# help you for debugging weird behavior. (Consider this seriously if you
# are having debugging problems.)

# def save_feedforward_classifier(filename, feedforward_classifier_state, feedforward_classifier_connections):
#     None
# def load_feedforward_classifier(filename):
#     return [feedforward_classifier_state, feedforward_classifier_connections]

# def save_autoencoder(filename, autoencoder_state, autoencoder_connections):
#     None
# def load_autoencoder(filename):
#     return [autoencoder_state, autoencoder_connections]

# def save_autoencoder_classifier(filename, autoencoder_classifier_state, autoencoder_classifier_connections):
#     None
# def load_autoencoder(filename):
#     return [autoencoder_classifier_state, autoencoder_classifier_connections]



if __name__=='__main__':

    # Please use the following snippet for clarity if you have command-line 
    # arguments:

    # if len(argv) < <number_of_expected_arguments>:
    #    print('Usage: python', argv[0], '<expected_arg_1>', '<expected_arg_2>, ...')
    #    exit(1)
    
    # arg_1 = argv[1]
    # arg_2 = argv[2]
    # ...
    
    
    # Read data here
    full_mnist_data, full_mnist_labels = mnist.read_mnist_training_data(train_size + test_size)
    training_data = [full_mnist_data[:train_size], full_mnist_labels[:train_size]]
    test_data = [full_mnist_data[train_size:], full_mnist_labels[train_size:]]
    
    # You may also use the gui_template.py functions to collect image data from the user. eg:
    # training_data = gt.get_images()
    # if len(training_data) != 0:
	#     gt.visualize_image(training_data[-1])
    
    # If you wish, you can have a loop here, or any other place really, that will update the
    # parameters below automatically.
    
    # Initialize network(s) here
    input_size = (28 * 28)  # Pixels in the image
    output_size = 10  # Possible classifications
    layer_sizes = np.asarray([input_size, np.sqrt(input_size), output_size])
    initialization_params = [layer_sizes]
    feedforward_classifier_state = None
    feedforward_classifier_connections = None 
    [feedforward_classifier_state, feedforward_classifier_connections] = init_feedforward_classifier(initialization_params)
    # Change initialization params if desired
    autoencoder_state = None
    autoencoder_connections = None
    [autoencoder_state, autoencoder_connections] = init_autoencoder(initialization_params)
    # Change initialization params if desired
    autoencoder_classifier_state = None
    autoencoder_classifier_connections = None
    [autoencoder_classifier_state, autoencoder_classifier_connections] = init_autoencoder_classifier(initialization_params)
    
    
    # Train network(s) here
    training_params = [training_runs]
    feedforward_classifier_connections = train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params)
    # Change training params if desired
    autoencoder_connections = train_autoencoder(autoencoder_state, autoencoder_connections, training_data, training_params)
    # Change training params if desired
    autoencoder_classifier_connections = train_autoencoder(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params)
   
    
    # Test network(s) here
    test_params = None
    test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data, test_params)
    # Change test params if desired
    test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params)
    # Change test params if desired
    test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params)
	
	# You can use gui_template.py functions for visualization as you wish
