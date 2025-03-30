from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the digits dataset
digits = np.load('DataSample.npy') #
np.random.shuffle(digits) #
X_train = digits[:50000,:-1] #  Feature
y_train = digits[:50000,-1] # Feature

RF = RandomForestClassifier()
grid_param = {'n_estimators': [400,450,500,550,600], # 50-100
              'max_depth': [30, 40, 50 , 60, 70, 80],
              'min_samples_split': [ 2,3]} # 0.1,0.3,0.5,0.8

#----------------------- Cross Validate----------------------------------
grid = GridSearchCV(RF, grid_param, cv=5)
grid.fit(X_train, y_train)

# best parameter combination
grid.best_params_

# Score achieved with best parameter combination
grid.best_score_

# all combinations of hyperparameters
grid.cv_results_['params']

# average scores of cross-validation
grid.cv_results_['mean_test_score']

print("Best: %f using %s" % (grid.best_score_,grid.best_params_)) #

means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))

