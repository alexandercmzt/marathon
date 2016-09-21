import numpy as np
from cross_validation import crossValidate
from classifiers import LinearRegressor as LR

# define data (need not be np arrays)
X = np.array([[0.86, 0.09,-0.85,0.87,-0.44,-0.43,-1.10,0.4,-.96,0.17]]).T
y = np.array([2.49,0.83,-0.25,3.1,0.87,0.02,-0.12,1.81,-0.83,0.43])

# define the number of partitions you want
n = 10

# define the model structures you want to test
structures = [
    {
        'powers': [[1]],
        'products': []
    },
    {
        'powers': [[1,2]],
        'products': []
    },
    {
        'powers': [[1,2,3]],
        'products': []
    },
    {
        'powers': [[1,2,3,4]],
        'products': []
    }
]

# define which regressor you want to use
regressor = LR

# cross validate to get the best model
crossValidate(X, y, structures, regressor, n)
