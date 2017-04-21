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

import time
t0 = time.time()
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
t1 = time.time()
import network
t2 = time.time()
net=network.Network([784, 30, 10])
t3 = time.time()
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
t4 = time.time()

print t1-t0
print t2-t1
print t3-t2
print t4-t3

Microsoft Windows [版本 6.1.7601]
Copyright (c) 2009 Microsoft Corporation.  All rights reserved.

D:\Users\IS_SM>e:

E:\>cd 2017-04-21-nndl

E:\2017-04-21-nndl>dir
 磁碟區 E 中的磁碟是 Backup
 磁碟區序號:  984B-6913

 E:\2017-04-21-nndl 的目錄

21/04/2017  15:44    <DIR>          .
21/04/2017  15:44    <DIR>          ..
21/04/2017  15:44    <DIR>          neural-networks-and-deep-learning-master
21/04/2017  15:40        36,704,509 neural-networks-and-deep-learning-master.zip

               1 個檔案      36,704,509 位元組
               3 個目錄  740,422,119,424 位元組可用

E:\2017-04-21-nndl>pip

Usage:
  pip <command> [options]

Commands:
  install                     Install packages.
  download                    Download packages.
  uninstall                   Uninstall packages.
  freeze                      Output installed packages in requirements format.
  list                        List installed packages.
  show                        Show information about installed packages.
  check                       Verify installed packages have compatible dependen
cies.
  search                      Search PyPI for packages.
  wheel                       Build wheels from your requirements.
  hash                        Compute hashes of package archives.
  completion                  A helper command used for command completion.
  help                        Show help for commands.

General Options:
  -h, --help                  Show help.
  --isolated                  Run pip in an isolated mode, ignoring
                              environment variables and user configuration.
  -v, --verbose               Give more output. Option is additive, and can be
                              used up to 3 times.
  -V, --version               Show version and exit.
  -q, --quiet                 Give less output. Option is additive, and can be
                              used up to 3 times (corresponding to WARNING,
                              ERROR, and CRITICAL logging levels).
  --log <path>                Path to a verbose appending log.
  --proxy <proxy>             Specify a proxy in the form
                              [user:passwd@]proxy.server:port.
  --retries <retries>         Maximum number of retries each connection should
                              attempt (default 5 times).
  --timeout <sec>             Set the socket timeout (default 15 seconds).
  --exists-action <action>    Default action when a path already exists:
                              (s)witch, (i)gnore, (w)ipe, (b)ackup, (a)bort.
  --trusted-host <hostname>   Mark this host as trusted, even though it does
                              not have valid or any HTTPS.
  --cert <path>               Path to alternate CA bundle.
  --client-cert <path>        Path to SSL client certificate, a single file
                              containing the private key and the certificate
                              in PEM format.
  --cache-dir <dir>           Store the cache data in <dir>.
  --no-cache-dir              Disable the cache.
  --disable-pip-version-check
                              Don't periodically check PyPI to determine
                              whether a new version of pip is available for
                              download. Implied with --no-index.

E:\2017-04-21-nndl>python
Python 2.7.12 |Anaconda 4.1.1 (32-bit)| (default, Jun 29 2016, 11:42:13) [MSC v.
1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://anaconda.org
>>> exit()

E:\2017-04-21-nndl>pip install -r requirements.txt
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

E:\2017-04-21-nndl>dir
 磁碟區 E 中的磁碟是 Backup
 磁碟區序號:  984B-6913

 E:\2017-04-21-nndl 的目錄

21/04/2017  15:45    <DIR>          .
21/04/2017  15:45    <DIR>          ..
08/02/2017  05:33                52 .gitignore
21/04/2017  15:44    <DIR>          data
21/04/2017  15:44    <DIR>          dennis-running-info
21/04/2017  15:44    <DIR>          fig
21/04/2017  15:44    <DIR>          need2move.no.underlines
21/04/2017  15:40        36,704,509 neural-networks-and-deep-learning-master.zip

08/02/2017  05:33             1,455 README.md
08/02/2017  05:33                36 requirements.txt
21/04/2017  15:44    <DIR>          src
               4 個檔案      36,706,052 位元組
               7 個目錄  740,422,119,424 位元組可用

E:\2017-04-21-nndl>cd src

E:\...(nndl_rootdir)...\src>dir
 磁碟區 E 中的磁碟是 Backup
 磁碟區序號:  984B-6913

 E:\...(nndl_rootdir)...\src 的目錄

21/04/2017  15:44    <DIR>          .
21/04/2017  15:44    <DIR>          ..
08/02/2017  05:33            12,662 conv.py
08/02/2017  05:33                46 dennis_test_module.py
08/02/2017  05:33             1,982 expand_mnist.py
08/02/2017  05:33               334 fibo.py
08/02/2017  05:33             2,673 mnist_average_darkness.py
08/02/2017  05:33             3,485 mnist_loader.py
08/02/2017  05:33               758 mnist_svm.py
08/02/2017  05:33             6,572 network.py
08/02/2017  05:33            14,296 network2.py
08/02/2017  05:33            12,945 network3.py
21/04/2017  15:44    <DIR>          old
08/02/2017  05:33                57 test-write.txt
              11 個檔案          55,810 位元組
               3 個目錄  740,422,119,424 位元組可用

E:\...(nndl_rootdir)...\src>dir
 磁碟區 E 中的磁碟是 Backup
 磁碟區序號:  984B-6913

 E:\...(nndl_rootdir)...\src 的目錄

21/04/2017  15:44    <DIR>          .
21/04/2017  15:44    <DIR>          ..
08/02/2017  05:33            12,662 conv.py
08/02/2017  05:33                46 dennis_test_module.py
08/02/2017  05:33             1,982 expand_mnist.py
08/02/2017  05:33               334 fibo.py
08/02/2017  05:33             2,673 mnist_average_darkness.py
08/02/2017  05:33             3,485 mnist_loader.py
08/02/2017  05:33               758 mnist_svm.py
08/02/2017  05:33             6,572 network.py
08/02/2017  05:33            14,296 network2.py
08/02/2017  05:33            12,945 network3.py
21/04/2017  15:44    <DIR>          old
08/02/2017  05:33                57 test-write.txt
              11 個檔案          55,810 位元組
               3 個目錄  740,422,119,424 位元組可用

E:\...(nndl_rootdir)...\src>python
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
>>> time()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'time' is not defined
>>> import time
>>> t0=time.time()
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
Epoch 0: 9451 / 10000
Epoch 1: 9501 / 10000
Epoch 2: 9473 / 10000
Epoch 3: 9484 / 10000
Epoch 4: 9443 / 10000
Epoch 5: 9477 / 10000
Epoch 6: 9459 / 10000
Epoch 7: 9492 / 10000
Epoch 8: 9490 / 10000
Epoch 9: 9490 / 10000
Epoch 10: 9470 / 10000
Epoch 11: 9491 / 10000
Epoch 12: 9488 / 10000
Epoch 13: 9491 / 10000
Epoch 14: 9484 / 10000
Epoch 15: 9479 / 10000
Epoch 16: 9485 / 10000
Epoch 17: 9468 / 10000
Epoch 18: 9503 / 10000
Epoch 19: 9467 / 10000
Epoch 20: 9486 / 10000
Epoch 21: 9481 / 10000
Epoch 22: 9473 / 10000
Epoch 23: 9484 / 10000
Epoch 24: 9451 / 10000
Epoch 25: 9483 / 10000
Epoch 26: 9465 / 10000
Epoch 27: 9458 / 10000
Epoch 28: 9473 / 10000
Epoch 29: 9481 / 10000
>>> t1=time.time()
>>> print t1 - t0
220.017999887
>>> exit()

E:\...(nndl_rootdir)...\src>python dennis-run.py
Epoch 0: 9091 / 10000
Epoch 1: 9267 / 10000
Epoch 2: 9323 / 10000
Epoch 3: 9377 / 10000
Epoch 4: 9396 / 10000
Epoch 5: 9394 / 10000
Epoch 6: 9432 / 10000
Epoch 7: 9430 / 10000
Epoch 8: 9453 / 10000
Epoch 9: 9446 / 10000
Epoch 10: 9475 / 10000
Epoch 11: 9457 / 10000
Epoch 12: 9458 / 10000
Epoch 13: 9450 / 10000
Epoch 14: 9452 / 10000
Epoch 15: 9468 / 10000
Epoch 16: 9484 / 10000
Epoch 17: 9457 / 10000
Epoch 18: 9483 / 10000
Epoch 19: 9495 / 10000
Epoch 20: 9497 / 10000
Epoch 21: 9515 / 10000
Epoch 22: 9488 / 10000
Epoch 23: 9512 / 10000
Epoch 24: 9494 / 10000
Epoch 25: 9501 / 10000
Epoch 26: 9495 / 10000
Epoch 27: 9497 / 10000
Epoch 28: 9515 / 10000
Epoch 29: 9493 / 10000
3.54200005531
0.00699996948242
0.000999927520752
211.679000139

E:\...(nndl_rootdir)...\src>python dennis-run.py > dennis-run-4.txt

  
Also read for more details:  
  
http://deeplearning.net/tutorial/lenet.html
  https://github.com/mdenil/dropout
    https://github.com/mdenil/dropout
