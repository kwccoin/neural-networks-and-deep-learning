# cd this directory
# # python (assume 2.x)
# python dennis-run.py > dennis-run.txt

print "dennis-run starting ..."
import time
t0 = time.time()
#print "mnist_loader starting ..."
#import mnist_loader
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data =[ [[ 0.        ],[ 0.        ]],[[1]]]
validation_data =[ [[ 0.        ],[ 0.        ]],[[1]]]
test_data =[ [[ 0.        ],[ 0.        ]],1]

print "training_data : {0}".format(training_data[0])
print "validation_data : {0}".format(validation_data[0])
print "test_data : {0}".format(test_data[0])


print "dennis-run program starting"
t1 = time.time()

import network_rodeo
#import network3a
t2 = time.time()
net=network_rodeo.Network([784, 2, 10])

print "net.num_layers : {0}".format(net.num_layers)
print "net.sizes : {0}".format(net.sizes)
print "net.biases : {0}".format(net.biases) 
print "net.weights : {0}".format(net.weights)

t3 = time.time()
print "dennis-run training starting"

#  def SGD(self, training_data, epochs, mini_batch_size, eta,
#            test_data=None):

#net.SGD(training_data, 2, 1, 3.0, test_data=test_data)

net.SGD(NOR_training_data, 2, 1, 1.0) #, test_data=NOR_test_data)


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







