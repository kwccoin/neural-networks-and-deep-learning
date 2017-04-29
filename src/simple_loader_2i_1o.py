# 2 input and 1 output

import numpy as np
test_data  =      [(np.array([[ 0.        ], [ 0.        ]]), 1), 
			       (np.array([[ 0.        ], [ 1.        ]]), 0), 
			       (np.array([[ 1.        ], [ 0.        ]]), 0), 
			       (np.array([[ 1.        ], [ 1.        ]]), 1)]

#import numpy as np
training_data = [ (np.array([[ 0.        ], [ 0.        ]]), np.array([[ 1.]])), 
                  (np.array([[ 0.        ], [ 1.        ]]), np.array([[ 0.]])), 
                  (np.array([[ 1.        ], [ 0.        ]]), np.array([[ 0.]])), 
                  (np.array([[ 1.        ], [ 1.        ]]), np.array([[ 1.]]))]


#import numpy as np
validation_data = [(np.array([[ 0.        ],[ 0.        ]]), 1), 
  				   (np.array([[ 0.        ],[ 1.        ]]), 0), 
  				   (np.array([[ 1.        ],[ 0.        ]]), 0),  
  				   (np.array([[ 1.        ],[ 1.        ]]), 1)]