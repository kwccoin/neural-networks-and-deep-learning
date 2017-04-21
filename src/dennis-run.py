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
import network
t2 = time.time()
net=network.Network([784, 30, 10])
t3 = time.time()
print "dennis-run training starting"
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
t4 = time.time()
print "dennis-run training end"

print t1-t0
print t2-t1
print t3-t2
print t4-t3

exit()

# type dennis-run.txt
# cat  dennis-run.txt

