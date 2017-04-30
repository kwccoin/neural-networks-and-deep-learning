"""
network_rodeo.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # np.random.seed(1)
            # dennis fix the randomness in output :-)
            # try on the running readme.py first
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        print "\n input a is  : {0}".format(a)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        print "output a is : {0}\n".format(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            #print "mini_batches : {0}".format(mini_batches)
            for mini_batch in mini_batches:
            	#print "mini_batch : {0}".format(mini_batch)
                self.update_mini_batch(mini_batch, eta)
                #import time
                #print "starting to sleep, please cancel"
                #time.sleep(500)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate3(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # dennis:
            # the key is that the self.weights and biases are NOT updated
            # hence during a mini_batch whilst the delta is accumulative
            # the forward calculation are not changed
            # effectively a whole "batch" is used to average out the changes
            # and 1 batch 1 new sets of weights and biases
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        print "\nself.biases : {0}".format(self.biases)
        print "self.weights : {0}\n".format(self.weights)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate1(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        dennis: for NOR gate it is not actually 2 h 1 but 2 h 2

        yes it look like this and hence logicall it is 2 h 1

         0 0 -> 1
         1 0 -> 0
         0 0 -> 0
         1 1 -> 1

         but if you use only 1 neuron at the back one has to change the program
         or in our case use 0 for 0 and 1 for 1 but really require change of program

         This is a need to do so if one simulate 0-9 digit
         using 2^4 i.e. 0000 0001 0010 0011 ... 0110

         That also require change of program actually you cannot just use an index
         you have to use an array as well for the validation ...

         2 h ... h 2 (not 1)

         (0 0) -> (0, 1)
         (1 0) -> (1, 0)
         (0 1) -> (1, 0)
         (1 1) -> (0, 1)

         Now when input testing and the number is an index ... it is classifer

         (0 0) -> 1
         (1 0) -> 0
         (0 1) -> 0
         (1 1) -> 1

         Here we try to to try the regression appraoch i.e. cf the value


        """

        # if not using index, one can loop and compare index
        # assume testing, validiation and training same data structure
        # after feedforward one can compare the patten of self.forward and y
        # and if agree update the test_results not by argmax but loop

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def evaluate2(self, test_data):
        test_results = [self.feedforward(x) - y
                        for (x, y) in test_data]

        #print "test_results : {0}".format(test_results)

        #test_results : [array([[ 0.],[ 0.]]),
        #				array([[-1.],[ 1.]]),
        #				array([[-1.],[ 1.]]),
        #	   			array([[ 0.],[ 0.]])]

        # how to do sum of mean square error to check the progress of the epochs

        # i.e. how to get mse which I think is
        # ((0**2 + 0**2) + (-1**2 + 1**2) + (-1**2 + 1**2) + (0**2 + 0**2) ) / 4

        sumarray = 0
        i = 0

        print "test_results : {0}".format(test_results)

        for arrays in test_results:
            for arrayi in arrays:
                # need to use TextWrangler to see invisiable
                #print
                #print "\nbefore"
                #print "------"
                #print "arrayi : {0}".format(arrayi)
                #print "sum(arrayi) : {0}".format(sum(arrayi))
                #print "i : {0}".format(i)
                sumarray = sumarray + np.sum(arrayi**2)
                i = i + 1
                #print "after"
                #print "------"
                #print "arrayi : {0}".format(arrayi)
                #print "sum(arrayi) : {0}".format(sum(arrayi))
                #print "i : {0}\n".format(i)

        # return sum
        print "sumarray : {0}".format(sumarray)

        print "i : {0}".format(i)

        print "(sumarray / i) : {0} ".format((sumarray / i))

        return (sumarray / i) # should divide by i...

    def evaluate3(self, test_data):

        # http://stackoverflow.com/questions/43705121/python-numpy-mean-of-square-calculation-is-this-the-right-way

        #mse = ((predictedMatrix - actualMatrix) ** 2).mean(axis=_axis)
        #_axis = 0 => Row wise computation to get a vector.
        #_axis = 1 => Column wise computation to get a vector.
        #_axis = None => Element wise computation to get a single number.

        for (x, y) in test_data:
            predictedMatrix = self.feedforward(x)
            actualMatrix    = y

        #print "predictedMatrix : {0}".format(predictedMatrix)

        #print "actualMatrix  : {0}".format(actualMatrix)


#        Waxis=None

        results = ((predictedMatrix - actualMatrix) ** 2).mean()

        #print "results mean() : {0}".format(results)

#        results = ((predictedMatrix - actualMatrix) ** 2).mean(axis=None)

#        print "results mean(axis=None) : {0}".format(results)

#        results = ((predictedMatrix - actualMatrix) ** 2).mean(Waxis)

#        print "results mean(Waxis) : {0}".format(results)

        return(results)


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
