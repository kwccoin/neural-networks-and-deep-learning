import numpy as np

# cd this directory
# # python (assume 2.x)
# python dennis-run.py > dennis-run.txt

x = np.array([ [[000,001,002],[010,011,012]], [[100,101,102],[110,111,112]], [[200,101,102],[210,111,112]], [[300,101,102],[310,111,112]] ] )
#print x[0]
#print x[1]
#print x[0,0]
#print x[1,1]
#print x[0,0,0]
#print x[1,1,1]

a = np.array( [ [[000,001,002],[010,011,012]], 
				[[100,101,102],[110,111,112]], 
				[[200,101,102],[210,111,112]], 
				[[300,101,102],[310,111,112]]  ])
				
b = np.array( [ [[000,001,002],[010,011,012]], [[100,101,102],[110,111,112]] ]) # from b
c = a[0:1]
d = a[0:2] # why it is d not c to make an array like b?

# That : is not actually range of 0 to 1 but a slice operation
# very complicated actually

print "\n\n 0 becomes octet ignore it\n\n"
print "a.shape : {0} \n a : {1}\n\n".format(a.shape, a) # (4, 2, 3)
print "b.shape : {0} \n b : {1}\n\n".format(b.shape, b) # (2, 2, 3)
print "c.shape : {0} \n c : {1}\n\n".format(c.shape, c) # (1, 2, 3)
print "d.shape : {0} \n d : {1}\n\n".format(d.shape, d) # (2, 2, 3)

#How can I form a np.array of 

x = np.array([ [[000,001,002],[010,011,012]], [[100,101,102],[110,111,112]], [[200,101,102],[210,111,112]], [[300,101,102],[310,111,112]]] )

y = np.array([ [[000,001,002],[010,011,012]], [[100,101,102],[110,111,112]] ])

# z = something from x and same as y

print "x : {0}".format(x)
print "y : {0}".format(y)
# print "z : {0}".format(z)

print "dennis-run starting ..."
import time
t0 = time.time()
print "mnist_loader starting ..."
import mnist_loader
#import mnist_loader_only2
training_data_full, validation_data_full, test_data_full = mnist_loader.load_data_wrapper()

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

training_data = training_data_full[0:2]
#import mnist_training_2_only
#training_data = mnist_training_2_only.training_data_2_only 

validation_data = validation_data_full[0:2]
import mnist_validation_2_only
validation_data = mnist_validation_2_only.validation_data_2_only 

test_data       = test_data_full[0:2]
import mnist_test_2_only
validation_data = mnist_test_2_only.test_data_2_only 


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


print "dennis-run program starting"
t1 = time.time()

import network_rodeo
#import network3a
t2 = time.time()
net=network_rodeo.Network([784, 1, 10])
#net=network_rodeo.Network([2, 2, 1])


print "net.num_layers : {0}".format(net.num_layers)
print "net.sizes : {0}".format(net.sizes)
print "net.biases : {0}".format(net.biases) 
print "net.weights : {0}".format(net.weights)

t3 = time.time()
print "dennis-run training starting"

#  def SGD(self, training_data, epochs, mini_batch_size, eta,
#            test_data=None):

#net.SGD(training_data, 2, 1, 3.0, test_data=test_data)

net.SGD(training_data, 2, 1, 0.5, test_data=test_data)


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









