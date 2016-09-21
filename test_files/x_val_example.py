from cross_validation import crossValidate
from classifiers import LogisticRegressor

# Example for two features x0 and x1
X = [[0,12],[1,10],[2,13],[100,50],[88,43],[90,54]]
y = [0,0,0,1,1,1]
# Here is an example of you define the model (set of features, powers, and products) you want to test
# powers = a list, where the entry at index i indicates the powers of feature i that we would like considered
#   so, here, we want to use the features x0, x0^2, x1
#   the length of this list must equal the number of features
#   if you want to leave a feature out, just pass [] at its index
# products = a list of lists, where each list numbers the features you want in the given product term
#   so, here, we want the feature x0x1
# To summarize, m1 says that we want to test the following feature set: [x0, x0^2, x1, x0x1]
m1 = {
    'powers': [[1, 2],[1]],
    'products': [[0,1]]
}
modelStructures = [m1]
crossValidate(X, y, modelStructures, LogisticRegressor, 2)
