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


# Allow switching nonlinearities
def tanh_derivative(x):
    return 1 - np.square(np.tanh(x))
nonlinear = np.tanh
nonlinear_derivative = tanh_derivative

# Initialize the corresponding networks
def init_feedforward_classifier(initialization_params):
    # Extract parameters
    layer_sizes = initialization_params[0]
    num_layers = layer_sizes.shape[0]
    max_layer_size = layer_sizes.max()

    # Initially, no neurons are firing, so state is all zeros
    feedforward_classifier_state = np.zeros((num_layers, max_layer_size))

    # Initialize classifier connections.  Use a 3D array, layers x inputs x outputs.
    # Also set connections for neurons which do not exist to 0
    feedforward_classifier_connections = np.random.random((num_layers - 1, max_layer_size, max_layer_size))
    for l in np.arange(layer_sizes.shape[0] - 1):
        feedforward_classifier_connections[l][layer_sizes[l]:max_layer_size] = 0
        feedforward_classifier_connections[l][:][layer_sizes[l+1]:max_layer_size] = 0

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
    for l in np.arange(1, feedforward_classifier_state.shape[0]):
        feedforward_classifier_state[l] = nonlinear(feedforward_classifier_connections[l-1]
                                                    .dot(feedforward_classifier_state[l-1]))
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

    # Stochastic gradient descent
    for i in np.arange(num_runs):
        rand_index = np.random.randint(0, data.shape[0])
        datum = data[rand_index]
        label = labels[rand_index]
        feedforward_classifier_state[0] = datum
        update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        feedforward_classifier_connections = backpropagate_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, label)

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

    return feedforward_classifier_connections

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
    full_mnist_data, full_mnist_labels = mnist.read_mnist_training_data(10000)
    training_data = [full_mnist_data[:5000], full_mnist_labels[:5000]]
    test_data = [full_mnist_data[5000:], full_mnist_labels[5000:]]
    
    # You may also use the gui_template.py functions to collect image data from the user. eg:
    # training_data = gt.get_images()
    # if len(training_data) != 0:
	#     gt.visualize_image(training_data[-1])
    
    # If you wish, you can have a loop here, or any other place really, that will update the
    # parameters below automatically.
    
    # Initialize network(s) here
    input_size = 28 * 28  # Pixels in the image
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
    training_params = None
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
