# cd this directory
# # python (assume 2.x)
# python dennis-run.py > dennis-run.txt

print "dennis-run starting ..."
import time
t0 = time.time()
print "mnist_loader starting ..."
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "dennis-run program starting"
t1 = time.time()
#import network
import network3a
t2 = time.time()
#net=network.Network([784, 30, 10])
t3 = time.time()
print "dennis-run training starting"
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#import network3
from network3a import Network
from network3a import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3a.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)

t4 = time.time()
print "dennis-run training end"

print t1-t0
print t2-t1
print t3-t2
print t4-t3

exit()

# type dennis-run.txt
# cat  dennis-run.txt
