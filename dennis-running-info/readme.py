#dennis-trying

# http://neuralnetworksanddeeplearning.com/chap1.html

# http://stackoverflow.com/questions/15821121/whats-the-working-directory-when-using-idle

##import os
##os.getcwd()
##
##os.chdir("/Users/dngMBA13user/Library/Mobile Documents/M6HJR9W95L~com~textasticapp~textastic/Documents/icloud-personal/github/neural-networks-and-deep-learning/src")
##os.getcwd()
##
##with open('test-write.txt', 'w+') as f:
##        f.write('This should be at C:\\Users\\poke\\Desktop\\someFile.txt now.\n'+
##                "/Users/dngMBA13user/Library/Mobile Documents/M6HJR9W95L~com~textasticapp~textastic/Documents/icloud-personal/github/neural-networks-and-deep-learning/src")
##f.close()
##
### not working
###f2 = open('test-write.txt','r+')
###f.read()
###f.close()
import numpy as np


import os
#os.chdir("/Users/dngMBA13user/Library/Mobile Documents/M6HJR9W95L~com~textasticapp~textastic/Documents/icloud-personal/github/neural-networks-and-deep-learning/src")
os.chdir("/Users/ext2int2adm/git-here/neural-networks-and-deep-learning/src")
print os.getcwd()
print os.listdir(os.curdir)
np.random.seed(1)

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
