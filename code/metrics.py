from sklearn.metrics import classification_report as report
from skimage.feature import hog
from utils import split_data, load_data, load_models, Config
from datetime import datetime

args = Config().get_config()

formatting = args['formatting']

input_path = args['input_path'] + '.npy'
crack_path = args['crack_path'] + '.npy'
pothole_path = args['pothole_path'] + '.npy'

svmcrack_path = args['svmcrack_path']
svmhole_path = args['svmhole_path']
crackpipe_path = args['crackpipe_path']
holepipe_path = args['holepipe_path']
enscrack_path = args['enscrack_path']
enshole_path = args['enshole_path']

X, crack, pothole = load_data(input_path,crack_path,pothole_path)
svmcrack, svmhole, crackpipe, holepipe, enscrack, enshole = load_models(
    svmcrack_path,svmhole_path,crackpipe_path,holepipe_path,enscrack_path,enshole_path
    )

## Split crack data
_, X_test, _, y_test = split_data(X, crack)

print(f'({datetime.now().strftime(formatting)}) Calculating metrics crack logistic classifier')
y_pred = crackpipe.predict(X_test)
prompt = 'Crack logistic classifier\n'
metrics = prompt + report(y_test,y_pred,target_names=['no_crack','crack'])
parameters = prompt + str(crackpipe.named_steps['lr'].get_params())

print(f'({datetime.now().strftime(formatting)}) Calculating metrics crack support vector classifier')
y_pred = svmcrack.predict(X_test)
prompt = '\n\nCrack support vector classifier\n'
metrics = metrics + prompt + report(y_test,y_pred,target_names=['no_crack','crack'])
parameters = parameters + prompt + str(svmcrack.get_params())

print(f'({datetime.now().strftime(formatting)}) Calculating metrics crack ensemble classifier')
y_pred = enscrack.predict(X_test)
prompt = '\n\nCrack voting classifier\n'
metrics = metrics + prompt + report(y_test,y_pred,target_names=['no_crack','crack'])

## Split pothole data
_, X_test, _, y_test = split_data(X, pothole)

print(f'({datetime.now().strftime(formatting)}) Calculating metrics pothole logistic classifier')
y_pred = holepipe.predict(X_test)
prompt = '\n\nLogistic pothole classifier\n'
metrics = metrics + prompt + report(y_test,y_pred,target_names=['no_pothole','pothole'])
parameters = parameters + prompt + str(holepipe.named_steps['lr'].get_params())

print(f'({datetime.now().strftime(formatting)}) Calculating metrics pothole support vector classifier')
y_pred = svmhole.predict(X_test)
prompt = '\n\nPothole support vector classifier\n'
metrics = metrics + prompt + report(y_test,y_pred,target_names=['no_pothole','pothole'])
parameters = parameters + prompt + str(svmhole.get_params())

print(f'({datetime.now().strftime(formatting)}) Calculating metrics pothole ensemble classifier')
y_pred = enshole.predict(X_test)
prompt = '\n\nPothole voting classifier\n'
metrics = metrics + prompt + report(y_test,y_pred,target_names=['no_pothole','pothole'])


## Save classification reports
with open('../results/metrics.txt','w') as file:
    file.write(metrics)

## Save parameters file
with open('../results/parameters.txt','w') as file:
    file.write(parameters)

