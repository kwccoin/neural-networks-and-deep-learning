import numpy as np

# cd this directory
# # python (assume 2.x)
# python dennis-run.py > dennis-run.txt

print "dennis-run starting ..."
import time
t0 = time.time()

print "simple_regression_2i2o_loader ..."
#import simple_loader_2i_1o #<- 2i * 1o
import simple_regression_2i2o_loader  #<- 2i * 2o
training_data   = simple_regression_2i2o_loader.training_data
validation_data = simple_regression_2i2o_loader.validation_data
test_data       = simple_regression_2i2o_loader.test_data


#print "mnist_loader starting ..."
#import mnist_loader
#import mnist_loader_only2
#training_data_full, validation_data_full, test_data_full = mnist_loader.load_data_wrapper()

#print "==== start list has no shape cannot ... "
#print "training_data full shape : {0}".format(training_data_full.shape) # [0])
#print "===="
#print "validation_data full shape : {0}".format(validation_data_full.shape) #[0])
#print "===="
#print "test_data shape full : {0}".format(test_data_full.shape) #[0])
#print "==== end"


#training_data =[ [[ 0.        ],[ 0.        ]],[[1]]]
#validation_data =[ [[ 0.        ],[ 0.        ]],[[1]]]
#test_data =[ [[ 0.        ],[ 0.        ]],1]

#training_data   = training_data_full[0:2]

#training_data = training_data_full[0:2]
#import mnist_training_2_only
#training_data = mnist_training_2_only.training_data_2_only

#validation_data = validation_data_full[0:2]
#import mnist_validation_2_only
#validation_data = mnist_validation_2_only.validation_data_2_only

#test_data       = test_data_full[0:2]
#import mnist_test_2_only
#validation_data = mnist_test_2_only.test_data_2_only


print "==== start"
#print "training_data shape : {0}".format(training_data.shape) # [0])
print "training_data : {0}".format(training_data) # [0])
print "===="
#print "validation_data shape : {0}".format(validation_data.shape) #[0])
print "validation_data : {0}".format(validation_data) # [0])
print "===="
#print "test_data shape : {0}".format(test_data.shape) #[0])
print "test_data 0 : {0}".format(test_data) # [0])
print "==== end"


print "simple_regression_2i2o_network starting"
t1 = time.time()

import simple_regression_2i2o_network
#import network3a
t2 = time.time()
#net=network_rodeo.Network([784, 1, 10])
net=simple_regression_2i2o_network.Network([2, 2, 2]) # due to 1 bin
#never use this - 2 2 1

#net.num_layers : 3
#net.sizes : [2, 2, 2] <--- not sure how it works
#net.biases = [np.array([[ 0.11],
#       [ 0.12]]), np.array([[ 0.21],
#       [0.22 ]])]
#net.weights = [np.array([[ 0.31, 0.32],
#       [0.42,  0.42]]), np.array([[ 0.51,  0.52],
#       [0.61,  0.62]])]

# now some of those 2-h2-2 generate 0.05 0.99 ... not sure how to do that !!!
# how to test that ... not sure

# from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# not sure but in general, weight and bias should belong to the layer n not n-1
# also we have same bias ...

# the format is generated after one run and change : to = and array to np.array

net.biases = [np.array([[0.35],[ 0.35]]),
              np.array([[0.60],[ 0.60]])]

net.weights = [np.array([[ 0.15 ,  0.20],
                      [ 0.25,   0.30]]),
               np.array([[ 0.40,   0.45],
                      [ 0.50,   0.55]])]

# after first running the weights and bias is ok but not the calculation of error ?

#self.biases : [array([[ 0.34570531],
#       [ 0.34526673]]), array([[ 0.53157333],
#       [ 0.6176443 ]])]
#self.weights : [array([[ 0.14978527,  0.19570531],
#       [ 0.24976334,  0.29526673]]), array([[ 0.35648927,  0.4048506 ],
#       [ 0.51121955,  0.5616421 ]])]

# cf with

# w_1^{+} = 0.149780716
# w_2^{+} = 0.19956143
# w_3^{+} = 0.24975114
# w_4^{+} = 0.29950229

# and




print "\nnet.num_layers : {0}".format(net.num_layers)
print "net.sizes : {0}".format(net.sizes)
print "net.biases : {0}".format(net.biases)
print "net.weights : {0}\n".format(net.weights)



t3 = time.time()
print "dennis-run training starting"

#  def SGD(self, training_data, epochs, mini_batch_size, eta,
#            test_data=None):

#net.SGD(training_data, 2, 1, 3.0, test_data=test_data)

# good for both simple classifer and regression
#net.SGD(training_data, 1000, 100, 2.0, test_data=test_data)

# just for comparsion with
#net.SGD(training_data, 1, 1, 0.5, test_data=test_data)

net.SGD(training_data, 500, 10, 0.5, test_data=test_data)


#time.sleep(500)

#import network3
#from network3a import Network
#from network3a import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
#training_data, validation_data, test_data = network3a.load_data_shared()
#mini_batch_size = 10
#net = Network([
#        FullyConnectedLayer(n_in=784, n_out=100),
#        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1,
#            validation_data, test_data)

t4 = time.time()
print "dennis-run training end"

print t1-t0
print t2-t1
print t3-t2
print t4-t3

exit()

# type dennis-run.txt
# cat  dennis-run.txt
