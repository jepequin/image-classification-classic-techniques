import dill
import numpy as np

from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from utils import hogify, split_data, load_data, Config

args = Config().get_config()

formatting = args['formatting']

input_path = args['input_path'] + '.npy'
crack_path = args['crack_path'] + '.npy'
pothole_path = args['pothole_path'] + '.npy'

crackpipe_path = args['crackpipe_path']
holepipe_path = args['holepipe_path']
svmcrack_path = args['svmcrack_path']
svmhole_path = args['svmhole_path']
enscrack_path = args['enscrack_path']
enshole_path = args['enshole_path']

X, crack, pothole = load_data(input_path,crack_path,pothole_path)

## Parameter grid for logistic classifiers
log_grid = {'lr__C':np.arange(0.4, 1.4, 0.2)}

## Parameter grid for support vector classifiers
svm_grid = {'C':np.logspace(-2, 2, num=5, base=2)}

## Split crack data
X_train, X_test, y_train, y_test = split_data(X, crack)

################################
### Crack logistic regressor ###
################################
print(f'({datetime.now().strftime(formatting)}) Training crack logistic classifier')

lr = LogisticRegression(solver='liblinear', dual=True, max_iter=1000, class_weight='balanced') 
crackpipe = Pipeline([('hog',hogify),('lr',lr)])
crackgrid = GridSearchCV(crackpipe,param_grid=log_grid,verbose=2)
crackgrid.fit(X_train,y_train)
crackpipe = crackgrid.best_estimator_
dill.dump(crackpipe, open(crackpipe_path, 'wb'))
print(f"Pothole logistic classifier saved to '{crackpipe_path}'")

#################
### Crack svm ###
#################
print(f'({datetime.now().strftime(formatting)}) Training crack support vector classifier')

X_train, X_test, y_train, y_test = split_data(X, crack)

svm = SVC(class_weight='balanced',probability=True)
svmgrid = GridSearchCV(svm,svm_grid,verbose=2)
svmgrid.fit(X_train,y_train)
svmcrack = svmgrid.best_estimator_

dill.dump(svmcrack, open(svmcrack_path, 'wb'))
print(f"Crack support vector classifier saved to '{svmcrack_path}'")

#################################
### Crack ensemble classifier ###
#################################
print(f'({datetime.now().strftime(formatting)}) Fitting crack voting classifier')

estimators = [('crackpipe',crackpipe),('svmcrack',svmcrack)]
enscrack = VotingClassifier(estimators,voting='soft')
enscrack.fit(X_train,y_train)

dill.dump(enscrack, open(enscrack_path, 'wb'))
print(f"Ensemble classifier saved to '{enscrack_path}'")



## Split pothole data
X_train, X_test, y_train, y_test = split_data(X, pothole)

##################################
### Pothole logistic regressor ###
##################################
print(f'({datetime.now().strftime(formatting)}) Training pothole logistic regressor')

lr = LogisticRegression(solver='liblinear', dual=True, max_iter=1000, class_weight='balanced') 
holepipe = Pipeline([('hog',hogify),('lr',lr)])
holegrid = GridSearchCV(holepipe,log_grid,verbose=2)
holegrid.fit(X_train,y_train)
holepipe = holegrid.best_estimator_

dill.dump(holepipe, open(holepipe_path, 'wb'))
print(f"Pothole logistic classifier saved to '{holepipe_path}'")

###################
### Pothole svm ###
###################
print(f'({datetime.now().strftime(formatting)}) Training pothole support vector classifier')

svm = SVC(class_weight='balanced',probability=True)
svmgrid = GridSearchCV(svm,svm_grid,verbose=2)
svmgrid.fit(X_train,y_train)
svmhole = svmgrid.best_estimator_

dill.dump(svmhole, open(svmhole_path, 'wb'))
print(f"Pothole support vector classifier saved to '{svmhole_path}'")

###################################
### Pothole ensemble classifier ###
###################################
print(f'({datetime.now().strftime(formatting)}) Fitting pothole voting classifier')

estimators = [('holepipe',holepipe),('svmhole',svmhole)]
enshole = VotingClassifier(estimators,voting='soft')
enshole.fit(X_train,y_train)
dill.dump(enshole, open(enshole_path, 'wb'))
print(f"Ensemble classifier saved to '{enshole_path}'")