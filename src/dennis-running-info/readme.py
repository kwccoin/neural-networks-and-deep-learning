#dennis-trying 2 in windows 

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

E:\...(nndl_roottdir)...>python  # checking version
Python 2.7.12 |Anaconda 4.1.1 (32-bit)| (default, Jun 29 2016, 11:42:13) [MSC v.
1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://anaconda.org
>>> exit()

E:\...(nndl_roottdir)...>pip install -r requirements.txt
Requirement already satisfied: numpy in d:\anaconda2\lib\site-packages (from -r
requirements.txt (line 1))
Requirement already satisfied: scikit-learn in d:\anaconda2\lib\site-packages (f
rom -r requirements.txt (line 2))
Requirement already satisfied: scipy in d:\anaconda2\lib\site-packages (from -r
requirements.txt (line 3))
Requirement already satisfied: Theano in d:\anaconda2\lib\site-packages (from -r
 requirements.txt (line 4))
Requirement already satisfied: six>=1.9.0 in d:\anaconda2\lib\site-packages (fro
m Theano->-r requirements.txt (line 4))


E:...(in the src under nndl_rootdir)...>python
Python 2.7.12 |Anaconda 4.1.1 (32-bit)| (default, Jun 29 2016, 11:42:13) [MSC v.
1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://anaconda.org
>>> import mnist_loader
>>> training_data, validation_data, test_data = \
... mnist_loader.load_data_wrapper()
>>> import network
>>> net=network.Nework([784, 30, 10])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'Nework'
>>> net=network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
Epoch 0: 9016 / 10000
Epoch 1: 9189 / 10000
Epoch 2: 9276 / 10000
Epoch 3: 9305 / 10000
Epoch 4: 9340 / 10000
Epoch 5: 9354 / 10000
Epoch 6: 9388 / 10000
Epoch 7: 9407 / 10000
Epoch 8: 9424 / 10000
Epoch 9: 9400 / 10000
Epoch 10: 9431 / 10000
Epoch 11: 9441 / 10000
Epoch 12: 9444 / 10000
Epoch 13: 9469 / 10000
Epoch 14: 9457 / 10000
Epoch 15: 9447 / 10000
Epoch 16: 9457 / 10000
Epoch 17: 9474 / 10000
Epoch 18: 9481 / 10000
Epoch 19: 9456 / 10000
Epoch 20: 9464 / 10000
Epoch 21: 9484 / 10000
Epoch 22: 9486 / 10000
Epoch 23: 9462 / 10000
Epoch 24: 9484 / 10000
Epoch 25: 9457 / 10000
Epoch 26: 9479 / 10000
Epoch 27: 9482 / 10000
Epoch 28: 9468 / 10000
Epoch 29: 9488 / 10000
>>>

http://stackoverflow.com/questions/2866380/how-can-i-time-a-code-segment-for-testing-performance-with-pythons-timeit

import time

t0 = time.time()
code_block
t1 = time.time()

total = t1-t0
  
Also read for more details:  
  
http://deeplearning.net/tutorial/lenet.html
  https://github.com/mdenil/dropout
    https://github.com/mdenil/dropout
