The topic of this project is: FeedForward-Backpropagation Neural Network, Autoencoders.  I chose to use the MNIST database of handwritten
digits, located at http://yann.lecun.com/exdb/mnist/

The project directory contains the following items:

parameters			folder containing configuration files
parameters/samples		example configuration files
output				folder containing output from test runs
output/samples			example output files
report				folder containing latex files for the report
Log.rtf				a log of my thoughts as I was implementing the project.  NOT the report.  Included for interest only.
mnist_load_show.py		code for loading and displaying mnist images- not written by me
mnist_load_show.pyc		see above
project_1_template.py		project code, based on the provided template
README.txt			this document
Report.pdf			project report
train-images-idx3-ubyte		mnist images.  Note that I split this set of images into train and test images in my code.
train-labels-idx1-ubyte		mnist labels


This project is controlled with configuration files.  The program will search in the folder "parameters" for files named 
"parameters_#.txt", replacing the # with 1, then 2, then 3, etc...  This continues as long as new files with the expected
names are found.  Each file contains the parameters for one full run of initialization, training, and testing for each 
network.

Output from each run is printed to "output/output_#_<test_name>.txt".  Test name is a parameter in the configuration file
and # is the corresponding parameter file number.  

Configuration files follow the form:


learning_rate_test	test name
8000 			train size
8000 			test size
0.01 			learning rate
16000 			training runs
1			use tanh?
0			display autoencoder comparison images with matplotlib?
1			use stochasitic gradient descent? (batch gradient descent with full data set if false)
10			number of epochs (only used if not using stochasitic gradient descent)
1			initial weight setting (0 = all zero, 1 = random -1 to 1, 2 = random in literature supported optimal range)
0			skip training and testing the feedforward classifier
0			skip training and testing the autoencoder
0			skip training and testing the autoencoder classifier
1 300			feedforward classifier hidden layers (number of layers, then layer sizes, eg. 0 or 1 200 or 2 100 100)
1 100			autoencoder hidden layers (same format)
1 100			autoencoder classifier layers after the autoencoder (same format)
1 200			autoencoder classifier layers in the autoencoder (same format)


Text after the parameters values will be ignored.  For boolean values, 1 = true and 0 = false.

Output files follow the form:


Test started.

Feedforward classifier training set performance:
##% correct prediction.
## mean squared error

Autoencoder training set performance:
## mean squared error.

Training autoencoder classifier...
Autoencoder training set performance:
## mean squared error.

Feedforward classifier training set performance:
##% correct prediction.
## mean squared error

Autoencoder classifier training set performance:
##% correct prediction.
## mean squared error

Feedforward classifier test set performance:
##% correct prediction.
## mean squared error

Autoencoder test set performance:
## mean squared error.

Autoencoder classifier test set performance:
##% correct prediction.
## mean squared error

Test finished.


The important information here are the test set performance of each of the networks, found in the last three sections (Feedforward 
classifier test set performance, Autoencoder test set performance, Autoencoder classifier test set performance).  

To see examples of parameter files and output files, look in parameters/samples and output/samples.  The files in these folders 
were used for the test runs described in the report.

Please feel free to contact me with comments or questions at ddenis@connect.carleton.ca

