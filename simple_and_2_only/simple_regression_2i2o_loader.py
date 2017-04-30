# here we try regression i.e. the test/validation is really the same structure as the training

# trying to do mean square error


import numpy as np

# basic NOR gate

test_data  =      [(np.array([[ 0.        ], [ 0.        ]]), np.array([[ 0.],[ 1.]])),  # this is same data structure now
			       (np.array([[ 0.        ], [ 1.        ]]), np.array([[ 1.],[ 0.]])),
			       (np.array([[ 1.        ], [ 0.        ]]), np.array([[ 1.],[ 0.]])),
			       (np.array([[ 1.        ], [ 1.        ]]), np.array([[ 0.],[ 1.]]))]

#import numpy as np

training_data = [ (np.array([[ 0.        ], [ 0.        ]]), np.array([[ 0.],[ 1.]])),
                  (np.array([[ 0.        ], [ 1.        ]]), np.array([[ 1.],[ 0.]])),
                  (np.array([[ 1.        ], [ 0.        ]]), np.array([[ 1.],[ 0.]])),
                  (np.array([[ 1.        ], [ 1.        ]]), np.array([[ 0.],[ 1.]]))]


#import numpy as np
validation_data = [(np.array([[ 0.        ],[ 0.        ]]), np.array([[ 0.],[ 1.]])), # this is same data structure now
  				   (np.array([[ 0.        ],[ 1.        ]]), np.array([[ 1.],[ 0.]])),
  				   (np.array([[ 1.        ],[ 0.        ]]), np.array([[ 1.],[ 0.]])),
  				   (np.array([[ 1.        ],[ 1.        ]]), np.array([[ 0.],[ 1.]]))]

# trying https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

test_data  =      [(np.array([[ 0.05       ], [ 1.0        ]]), np.array([[ 0.01],[ 0.99]]))]  # this is same data structure now

#import numpy as np

training_data =   [(np.array([[ 0.05       ], [ 1.0        ]]), np.array([[ 0.01],[ 0.99]]))]

#import numpy as np

validation_data = [(np.array([[ 0.05       ], [ 1.0        ]]), np.array([[ 0.01],[ 0.99]]))] # this is same data structure now



# --- testing code ----

# send to http://stackoverflow.com/questions/43705121/python-numpy-mean-of-square-calculation-is-this-the-right-way
# to see anyone have better coding skill
# for the moment move the following code to network.py

# We should do self.feedforward(x) but for here just

#self_forward_x = np.array([[ 0.],[ 1.]])

#test_results = [self_forward_x - y
#                for (x, y) in test_data]

##print "test_results : {0}".format(test_results)

##test_results : [array([[ 0.],[ 0.]]),
##				array([[-1.],[ 1.]]),
##				array([[-1.],[ 1.]]),
##	   			array([[ 0.],[ 0.]])]

## how to do sum of mean square error to check the progress of the epochs

## i.e. how to get mse which I think is
## ((0**2 + 0**2) + (-1**2 + 1**2) + (-1**2 + 1**2) + (0**2 + 0**2) ) / 4

#sumarray = 0

#for arrays in test_results:
#	for arrayi in arrays:
#		#print "arrayi : {0}".format(arrayi)
#		#print "sum(arrayi) : {0}".format(sum(arrayi))
#		sumarray = sumarray + np.sum(arrayi**2)

## return sum
#print "sum : {0}".format(sumarray)

#return sum(int(x == y) for (x, y) in test_results)
