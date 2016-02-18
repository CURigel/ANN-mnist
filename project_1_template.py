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
from scipy.special import expit
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,stats
import sklearn.preprocessing as skpre
# Uncomment if you will use the GUI code:
# import gui_template as gt


#Parameters

# Allow switching nonlinearities
def tanh_derivative(x):
    return (np.square(np.tanh(x)) * -1) + 1


def sigm_derivative(x):
    sigm = expit(x)
    return sigm * ((sigm * -1) + 1)


def tanh_derivative_wrt_tanhx(x):
    return (np.square(x) * - 1) + 1


def sigm_derivative_wrt_sigmx(x):
    return x * ((x * -1) + 1)


# See http://deeplearning.net/tutorial/mlp.html for justification
def tanh_init_weight_max(layer_sizes):
    layer_weight_maxes = np.ndarray(layer_sizes.shape[0] - 1)
    for i in np.arange(layer_weight_maxes.shape[0]):
        layer_weight_maxes[i] = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
    return layer_weight_maxes


# See http://deeplearning.net/tutorial/mlp.html for justification
def sigm_init_weight_max(layer_sizes):
    return tanh_init_weight_max(layer_sizes) * 4


nonlinear = np.tanh
nonlinear_min_value = -1
nonlinear_derivative = tanh_derivative
nonlinear_derivative_wrt_nonlinear_x = tanh_derivative_wrt_tanhx
nonlinear_max_init_weight = tanh_init_weight_max

train_size = 8000
test_size = 8000
learning_rate = 0.01
training_runs = 16000
stochastic_gradient_descent = True
num_epochs = 5000
display_autoencoder_images = False

init_weight_setting = 2  # 0 = all zero, 1 = random, 2 = random within literature supported optimum range

skip_feedforward_classifier = False
skip_autoencoder = False
skip_autoencoder_classifier = False
feedforward_classifier_hidden_layers = [300]
autoencoder_hidden_layers = [200]
autoencoder_classifier_autoencoder_hidden_layers = [200]
autoencoder_classifier_classifier_hidden_layers = [100]

output_file = 'output_1.txt'
test_name = 'default'


# Initialize the corresponding networks
def init_feedforward_classifier(initialization_params):
    # Extract parameters
    layer_sizes = initialization_params[0]
    num_layers = layer_sizes.shape[0]

    # Initially, no neurons are firing, so state is all zeros
    # Except for thresholds, which we set to 1.
    neuron_states = []
    threshold_values = [1 for i in np.arange(num_layers - 1)]
    for l in np.arange(num_layers):
        neuron_states.append(np.zeros(layer_sizes[l]))

    feedforward_classifier_state = [neuron_states, threshold_values]

    # Calculate random weight interval.
    max_weights = np.zeros(layer_sizes.shape[0]) # Assume all zero to start
    if init_weight_setting == 1: # -1 to 1
        max_weights += 1
    if init_weight_setting == 2: # Literature supported optimum
        max_weights = nonlinear_max_init_weight(layer_sizes)
    min_weights = max_weights * -1

    # Initialize classifier weights.  Use a list of 2D ndarrays.
    # Threshold weights too, but only 1D ndarrays needed for those
    layer_weights = []
    threshold_weights = []
    for l in np.arange(num_layers - 1):
        layer_weights.append(np.random.uniform(low=min_weights[l], high=max_weights[l],
                                               size=(layer_sizes[l], layer_sizes[l+1])))
        threshold_weights.append(np.random.uniform(low=min_weights[l], high=max_weights[l],
                                                   size=layer_sizes[l+1]))
    feedforward_classifier_connections = [layer_weights, threshold_weights]

    return [feedforward_classifier_state, feedforward_classifier_connections]


def init_autoencoder(initialization_params):
    return init_feedforward_classifier(initialization_params)


def init_autoencoder_classifier(initialization_params):
    autoencoder_init_params = initialization_params[0]
    classifier_init_params = initialization_params[1]
    [autoencoder_state, autoencoder_connections] = init_autoencoder(autoencoder_init_params)
    [classifier_state, classifier_connections] = init_feedforward_classifier(classifier_init_params)
    return [[autoencoder_state, classifier_state], [autoencoder_connections, classifier_connections]]


# Given an input, these functions calculate the corresponding output to 
# that input.
def update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections):
    neuron_states = feedforward_classifier_state[0]
    threshold_values = feedforward_classifier_state[1]
    layer_weights = feedforward_classifier_connections[0]
    threshold_weights = feedforward_classifier_connections[1]

    for l in np.arange(1, len(neuron_states)):
        input_sum = layer_weights[l-1].T.dot(neuron_states[l-1])
        input_sum += threshold_values[l-1] * threshold_weights[l-1]
        neuron_states[l] = nonlinear(input_sum)

    return feedforward_classifier_state


def update_autoencoder(autoencoder_state, autoencoder_connections):
    return update_feedforward_classifier(autoencoder_state, autoencoder_connections)


def update_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections):
    return update_feedforward_classifier(autoencoder_classifier_state, autoencoder_classifier_connections)
    

# Main functions to handle the training of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the update functions.
def train_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params):
    train_network(feedforward_classifier_state, feedforward_classifier_connections, training_data, training_params)
    output = open(output_file, 'a')
    output.write("Feedforward classifier training set performance:\n")
    output.close()
    output_feedforward_classifier_performance(feedforward_classifier_state, feedforward_classifier_connections, training_data)
    return feedforward_classifier_connections


def train_autoencoder(autoencoder_state, autoencoder_connections, training_data, training_params):
    train_network(autoencoder_state, autoencoder_connections, training_data, training_params)
    output = open(output_file, 'a')
    output.write("Autoencoder training set performance:\n")
    output.close()
    output_autoencoder_performance(autoencoder_state, autoencoder_connections, training_data)
    return autoencoder_connections


def train_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params):
    autoencoder_state = autoencoder_classifier_state[0]
    autoencoder_neuron_states = autoencoder_state[0]
    autoencoder_connections = autoencoder_classifier_connections[0]
    autoencoder_layer_weights = autoencoder_connections[0]
    autoencoder_threshold_weights = autoencoder_connections[1]
    classifier_state = autoencoder_classifier_state[1]
    classifier_neuron_states = classifier_state[0]
    classifier_connections = autoencoder_classifier_connections[1]
    classifier_layer_weights = classifier_connections[0]
    classifier_threshold_weights = classifier_connections[1]
    data = training_data[0]
    labels = training_data[1]
    autoencoder_training_data = [data, data]

    output = open(output_file, 'a')
    output.write('Training autoencoder classifier...\n')
    output.close()

    # First, we train the autoencoder.
    train_autoencoder(autoencoder_state, autoencoder_connections, autoencoder_training_data, training_params)

    # Now we map the training data to it's representation in the second autoencoder layer
    classifier_data = np.ndarray((data.shape[0], autoencoder_neuron_states[1].shape[0]))
    for i in np.arange(data.shape[0]):
        autoencoder_neuron_states[0] = data[i]
        update_autoencoder(autoencoder_state, autoencoder_connections)
        classifier_data[i] = autoencoder_neuron_states[1]

    classifier_training_data = [classifier_data, labels]
    # And use the newly mapped data to train the classifier
    train_feedforward_classifier(classifier_state, classifier_connections, classifier_training_data, training_params)

    # Now stitch the autoencoder and classifier together
    autoencoder_classifier_neuron_states = [autoencoder_neuron_states[0]] + classifier_neuron_states
    autoencoder_classifier_threshold_states = [autoencoder_state[1][0]] + classifier_state[1]
    autoencoder_classifier_layer_weights = [autoencoder_layer_weights[0]] + classifier_layer_weights
    autoencoder_classifier_threshold_weights = [autoencoder_threshold_weights[0]] + classifier_threshold_weights

    autoencoder_classifier_state = [autoencoder_classifier_neuron_states, autoencoder_classifier_threshold_states]
    autoencoder_classifier_connections = [autoencoder_classifier_layer_weights, autoencoder_classifier_threshold_weights]

    output = open(output_file, 'a')
    output.write('Autoencoder classifier training set performance:\n')
    output.close()
    output_feedforward_classifier_performance(autoencoder_classifier_state, autoencoder_classifier_connections,
                                              training_data)
    return [autoencoder_classifier_state, autoencoder_classifier_connections]


def train_network(state, connections, training_data, training_params):
    num_runs = training_params[0]
    neuron_states = state[0]
    num_outputs = neuron_states[-1].shape[0]
    data = training_data[0]
    labels = training_data[1]

    if stochastic_gradient_descent:
        for i in np.arange(num_runs):
            rand_index = np.random.randint(0, data.shape[0])
            connections = descend_point(data[rand_index], labels[rand_index], num_outputs, state, connections)
    else:  # Gradient descent
        for epoch in np.arange(num_epochs):
            for i in np.arange(data.shape[0]):
                connections = descend_point(data[i], labels[i], num_outputs, state, connections)


def descend_point(datum, label, num_outputs, feedforward_classifier_state, feedforward_classifier_connections):
    feedforward_classifier_state[0][0] = datum
    update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
    return backpropagate_feedforward_classifier(num_outputs, feedforward_classifier_state, feedforward_classifier_connections, label)


# Backpropagation functions
def backpropagate_feedforward_classifier(num_outputs, feedforward_classifier_state, feedforward_classifier_connections, label_vector):
    neuron_states = feedforward_classifier_state[0]
    num_layers = len(neuron_states)
    layer_weights = feedforward_classifier_connections[0]
    threshold_weights = feedforward_classifier_connections[1]

    weight_changes = []
    threshold_changes = []
    for l in np.arange(num_layers - 1):
        weight_changes.append(np.ndarray((layer_weights[l].shape[0], layer_weights[l].shape[1])))
        threshold_changes.append(np.ndarray(threshold_weights[l].shape[0]))

    # First consider the output deltas
    error_vector = np.zeros(neuron_states[-1].shape[0])
    error_vector = neuron_states[-1] - label_vector
    deriv_output_values = nonlinear_derivative_wrt_nonlinear_x(neuron_states[-1])
    output_delta = error_vector * deriv_output_values

    # Now loop over the rest of the layers
    # NOTE: W[i] goes from layer i to layer i+1, not from i-1 to i!
    prev_delta = output_delta
    for l in np.arange(0, num_layers-1)[::-1]:
        weight_changes[l] = -learning_rate * np.outer(neuron_states[l], prev_delta)
        threshold_changes[l] = -learning_rate * prev_delta
        weight_delta_sums = layer_weights[l].dot(prev_delta)
        prev_delta = nonlinear_derivative_wrt_nonlinear_x(neuron_states[l]) * weight_delta_sums

    for l in np.arange(num_layers - 1):
        layer_weights[l] += weight_changes[l]
        threshold_weights[l] += threshold_changes[l]
    return feedforward_classifier_connections


# Functions for outputting the results of an ANN on a data set
def output_feedforward_classifier_performance(feedforward_classifier_state, feedforward_classifier_connections, check_data):
    data = check_data[0]
    labels = check_data[1]
    neuron_states = feedforward_classifier_state[0]

    correct = 0
    total_error = 0.0
    for i in np.arange(data.shape[0]):
        neuron_states[0] = data[i]
        update_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections)
        prediction = np.argmax(neuron_states[-1])
        correct += 1 if prediction == np.argmax(labels[i]) else 0
        total_error += np.linalg.norm((neuron_states[-1] - labels[i]))

    output = open(output_file, 'a')
    output.write('{}% correct prediction.\n'.format((float(correct) / float(data.shape[0])) * 100))
    output.write('{} mean squared error\n\n'.format(total_error / float(data.shape[0])))
    output.close()


def output_autoencoder_performance(autoencoder_state, autoencoder_connections, check_data):
    data = check_data[0]
    labels = check_data[1]
    neuron_states = autoencoder_state[0]

    total_error = 0.0
    for i in np.arange(data.shape[0]):
        neuron_states[0] = data[i]
        update_autoencoder(autoencoder_state, autoencoder_connections)
        total_error += np.linalg.norm((neuron_states[-1] - labels[i]))

    output = open(output_file, 'a')
    output.write('{} mean squared error.\n\n'.format(total_error / float(data.shape[0])))
    output.close()

    # Show some pictures!
    if display_autoencoder_images:
        random_indices = np.random.randint(0, data.shape[0], 10)
        inputs = np.copy(data[random_indices])
        outputs = np.ndarray((10, data.shape[1]))
        for i in np.arange(10):
            neuron_states[0] = data[random_indices[i]]
            update_autoencoder(autoencoder_state, autoencoder_connections)
            outputs[i] = np.copy(neuron_states[-1])
        input_viewable = denormalize(inputs)
        output_viewable = denormalize(outputs)
        mnist.visualize(np.concatenate((input_viewable, output_viewable)))
    None


def denormalize(x):
    x_shifted = x - x.min()
    x_normed = x_shifted / x_shifted.max()
    x_scaled = x_normed * 255
    return x_scaled.astype(int)


# Main functions to handle the testing of the networks. 
# Feel free to write auxiliary functions and call them from here.
# These functions are supposed to call the 'run' functions.
def test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data, test_params):
    # We assume here that the network has already been trained.
    output = open(output_file, 'a')
    output.write("Feedforward classifier test set performance:\n")
    output.close()
    output_feedforward_classifier_performance(feedforward_classifier_state, feedforward_classifier_connections, test_data)
    None
    
def test_autoencoder(autoencoder_state, autoencoder_connections, test_data, test_params):
    # We assume here that the network has already been trained.
    output = open(output_file, 'a')
    output.write("Autoencoder test set performance:\n")
    output.close()
    output_autoencoder_performance(autoencoder_state, autoencoder_connections, test_data)
    None

def test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data, test_params):
    # We assume here that the network has already been trained.
    output = open(output_file, 'a')
    output.write("Autoencoder classifier test set performance:\n")
    output.close()
    output_feedforward_classifier_performance(autoencoder_classifier_state, autoencoder_classifier_connections, test_data)
    None


def label_vectors_from_indicies(indicies, size):
    error_vectors = np.tile(nonlinear_min_value, (indicies.shape[0], size))
    error_vectors[np.arange(indicies.shape[0]), indicies] = 1
    return error_vectors


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
def load_parameters(test_number):
    global output_file, train_size, test_size, learning_rate, training_runs, display_autoencoder_images, \
        stochastic_gradient_descent, num_epochs, init_weight_setting, skip_feedforward_classifier, skip_autoencoder, \
        skip_autoencoder_classifier, feedforward_classifier_hidden_layers, autoencoder_hidden_layers, \
        autoencoder_classifier_classifier_hidden_layers, autoencoder_classifier_autoencoder_hidden_layers, nonlinear, \
        nonlinear_min_value, nonlinear_derivative, nonlinear_derivative_wrt_nonlinear_x, nonlinear_max_init_weight

    file_name = 'parameters_' + str(test_number) + '.txt'
    if not os.path.isfile(file_name):
        return False
    param_file = open(file_name, 'r')

    def get_single_param(): return param_file.readline().split()[0]

    def get_list_param():
        line = param_file.readline().split()
        return [int(i) for i in line[1:int(line[0])+1]]

    test_name = get_single_param()
    output_file = 'out_{}_{}.txt'.format(test_number, test_name)
    train_size = int(get_single_param())
    test_size = int(get_single_param())
    learning_rate = float(get_single_param())
    training_runs = int(get_single_param())
    use_tanh = get_single_param() == '1'
    display_autoencoder_images = get_single_param() == '1'
    stochastic_gradient_descent = get_single_param() == '1'
    num_epochs = int(get_single_param())
    init_weight_setting = int(get_single_param()) # 0 = all zero, 1 = random, 2 = random within literature supported optimum range
    skip_feedforward_classifier = get_single_param() == '1'
    skip_autoencoder = get_single_param() == '1'
    skip_autoencoder_classifier = get_single_param() == '1'
    feedforward_classifier_hidden_layers = get_list_param()
    autoencoder_hidden_layers = get_list_param()
    autoencoder_classifier_classifier_hidden_layers = get_list_param()
    autoencoder_classifier_autoencoder_hidden_layers = get_list_param()

    nonlinear = np.tanh if use_tanh else expit
    nonlinear_min_value = -1 if use_tanh else 0
    nonlinear_derivative = tanh_derivative if use_tanh else sigm_derivative
    nonlinear_derivative_wrt_nonlinear_x = tanh_derivative_wrt_tanhx if use_tanh else sigm_derivative_wrt_sigmx
    nonlinear_max_init_weight = tanh_init_weight_max if use_tanh else sigm_init_weight_max
    param_file.close()
    return True


def main():
    test_number = 1

    while load_parameters(test_number):
        output = open(output_file, 'w')
        output.write("Test started.\n\n")
        output.close()

        # Read data here
        full_mnist_data, full_mnist_labels = mnist.read_mnist_training_data(train_size + test_size)
        training_data = [full_mnist_data[:train_size], full_mnist_labels[:train_size]]
        test_data = [full_mnist_data[train_size:], full_mnist_labels[train_size:]]

        # Vectorize labels
        training_data[1] = label_vectors_from_indicies(training_data[1], 10)
        test_data[1] = label_vectors_from_indicies(test_data[1], 10)

        # Normalize data
        training_data[0] = (training_data[0].astype(float) - training_data[0].mean()) / 255.0
        test_data[0] = (test_data[0].astype(float) - test_data[0].mean()) / 255.0

        # Modified data set for autoencoder
        autoencoder_training_data = [training_data[0], training_data[0]]
        autoencoder_test_data = [test_data[0], test_data[0]]

        # Initialize network(s) here
        input_size = (28 * 28)  # Pixels in the image
        output_size = 10  # Possible classifications
        layer_sizes = np.asarray([input_size] + feedforward_classifier_hidden_layers + [output_size])
        initialization_params = [layer_sizes]
        feedforward_classifier_state = None
        feedforward_classifier_connections = None
        [feedforward_classifier_state, feedforward_classifier_connections] = init_feedforward_classifier(
            initialization_params)

        # Change network shape for auto-encoder
        autoencoder_layer_sizes = np.asarray([input_size] + autoencoder_hidden_layers + [input_size])
        autoencoder_initialization_params = [autoencoder_layer_sizes]
        autoencoder_state = None
        autoencoder_connections = None
        [autoencoder_state, autoencoder_connections] = init_autoencoder(autoencoder_initialization_params)

        # Two sets of parameters for autoencoder classifier
        autoencoder_classifier_classifier_layer_sizes = np.asarray([autoencoder_classifier_autoencoder_hidden_layers[-1]] +
                                                                   autoencoder_classifier_classifier_hidden_layers +
                                                                   [output_size])
        autoencoder_classifier_autoencoder_layer_sizes = np.asarray([input_size] +
                                                                    autoencoder_classifier_autoencoder_hidden_layers +
                                                                    [input_size])
        autoencoder_classifier_init_params = [[autoencoder_classifier_autoencoder_layer_sizes],
                                              [autoencoder_classifier_classifier_layer_sizes]]
        autoencoder_classifier_state = None
        autoencoder_classifier_connections = None
        [autoencoder_classifier_state, autoencoder_classifier_connections] = init_autoencoder_classifier(
            autoencoder_classifier_init_params)

        # Train network(s) here
        training_params = [training_runs]
        if not skip_feedforward_classifier:
            feedforward_classifier_connections = train_feedforward_classifier(feedforward_classifier_state,
                                                                              feedforward_classifier_connections,
                                                                              training_data, training_params)
        if not skip_autoencoder:
            autoencoder_connections = train_autoencoder(autoencoder_state, autoencoder_connections,
                                                        autoencoder_training_data, training_params)
        if not skip_autoencoder_classifier:
            [autoencoder_classifier_state, autoencoder_classifier_connections] = train_autoencoder_classifier(
                autoencoder_classifier_state, autoencoder_classifier_connections, training_data, training_params)

        # Test network(s) here
        test_params = None
        if not skip_feedforward_classifier:
            test_feedforward_classifier(feedforward_classifier_state, feedforward_classifier_connections, test_data,
                                        test_params)
        if not skip_autoencoder:
            test_autoencoder(autoencoder_state, autoencoder_connections, autoencoder_test_data, test_params)
        if not skip_autoencoder_classifier:
            test_autoencoder_classifier(autoencoder_classifier_state, autoencoder_classifier_connections, test_data,
                                        test_params)
        output = open(output_file, 'a')
        output.write("Test finished.\n")
        output.close()
        test_number += 1


if __name__ == '__main__':
    main()
