# 2 input and 2 output
# the data structure is different and hence the checking program is not generic
# it assume binning perferctly
# no 0/1 representing 2 value
# no 4 bin represnting 10- 16 values
# one bin one value and no more than 1 bin!

import numpy as np
test_data  =      [(np.array([[ 0.        ], [ 0.        ]]), 1),  # this is index not actually 1
			       (np.array([[ 0.        ], [ 1.        ]]), 0),
			       (np.array([[ 1.        ], [ 0.        ]]), 0),
			       (np.array([[ 1.        ], [ 1.        ]]), 1)]

#import numpy as np

training_data = [ (np.array([[ 0.        ], [ 0.        ]]), np.array([[ 0.],[ 1.]])),
                  (np.array([[ 0.        ], [ 1.        ]]), np.array([[ 1.],[ 0.]])),
                  (np.array([[ 1.        ], [ 0.        ]]), np.array([[ 1.],[ 0.]])),
                  (np.array([[ 1.        ], [ 1.        ]]), np.array([[ 0.],[ 1.]]))]


#import numpy as np
validation_data = [(np.array([[ 0.        ],[ 0.        ]]), 1), # this is index not actually 1
  				   (np.array([[ 0.        ],[ 1.        ]]), 0),
  				   (np.array([[ 1.        ],[ 0.        ]]), 0),
  				   (np.array([[ 1.        ],[ 1.        ]]), 1)]
