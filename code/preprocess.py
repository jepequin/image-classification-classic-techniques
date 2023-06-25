import pandas as pd
import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray
from datetime import datetime
from utils import resizing, augment, Config

args = Config().get_config()

hog_bool = args['hog_bool']
augmentation = args['augmentation']

formatting = args['formatting']

csv_path = args['csv_path']
input_path = args['input_path']
crack_path = args['crack_path']
pothole_path = args['pothole_path']
data_dir = args['data_dir']

## Load data and resize inputs
print(f'({datetime.now().strftime(formatting)}) Loading data and resizing inputs ...')
df = pd.read_csv(csv_path)
X = df['image_name'].apply(resizing,data_dir=data_dir)
crack = df['crack']
pothole = df['pothole']

if augmentation:
	## Augmenting data
	print(f'({datetime.now().strftime(formatting)}) Augmenting data ...')
	X, crack, pothole = augment(X, crack, pothole)

## Grayify original images
X = X.apply(rgb2gray)

if hog_bool:
	## Calculate hog features
	print(f'({datetime.now().strftime(formatting)}) Calculating hog features ...')
	X = X.apply(hog, pixels_per_cell=(16,16), cells_per_block=(2,2))
else:
	## Flatten
	print(f'({datetime.now().strftime(formatting)}) Flattening images ...')
	X = X.apply(lambda img:img.flatten())

X = np.stack(X.to_numpy())
crack = crack.to_numpy()
pothole = pothole.to_numpy()

## Save data arrays
print(f'({datetime.now().strftime(formatting)}) Saving data arrays ...')
input_path = input_path + hog_bool*'_hog'
np.save(input_path,X)
np.save(crack_path,crack)
np.save(pothole_path,pothole)

print('Input and output numpy arrays successfully saved')


