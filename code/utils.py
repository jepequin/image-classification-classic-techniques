
import os
import yaml
import dill
import numpy as np
import functools
import pandas as pd

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import random_noise

## Image processing

def resizing(img_dir,data_dir=None):
	img_name = img_dir + '_RAW.jpg'
	img_path = os.path.join(data_dir,img_dir,img_name)
	img = imread(img_path)
	img = resize(img,(640,640),anti_aliasing=True) # anti_aliasing apply a Gaussian filter to smooth the image prior to downsampling
	return img

def flat_gray(func):
	@functools.wraps(func) # Preserve information about original function (e.g. __name__, __doc__)
	def wrapper(*args):
		flat_gray_lst = list() 
		for img in func(*args):
			flat_gray_lst.append(np.array(rgb2gray(img).flatten())) 
		return flat_gray_lst
	return wrapper

@flat_gray # Must feed 2D array when creating Series below 
def perturb(image):
	noised = random_noise(image, var=0.01**2)
	brighten = image + 50/255
	contrast = image * 1.1
	return (noised,brighten,contrast)

def augment(X,y,z):
	n_samples = len(X)
	X_augment = list()
	y_augment = list()
	z_augment = list()
	for idx in range(n_samples):
		augmented = perturb(X.iloc[idx])
		n_augmented = len(augmented)
		X_augment.extend(augmented)
		y_augment.extend([y.iloc[idx]]*n_augmented)
		z_augment.extend([z.iloc[idx]]*n_augmented)
	X_augment = pd.Series(X_augment)
	y_augment = pd.Series(y_augment)
	z_augment = pd.Series(z_augment)

	# Reshape augmented arrays
	X_augment = X_augment.apply(resize,output_shape=(640,640),anti_aliasing=True)

	# Concatenate original and augmented Series
	X = pd.concat([X,X_augment])
	y = pd.concat([y,y_augment])
	z = pd.concat([z,z_augment])
	return X, y, z

@FunctionTransformer
def hogify(imgs):
	# HOG naive algorithm:
	# 1. Decide number of bins (orientations) to divide the interval [0,180]
	# 2. Choose number of pixel to be grouped in same cell
	# 3. Choose number of cells to be grouped in same block
	# 4 For each pixel
	# 	a. Calculate magnitude and angle of gradient 
	# 	b. Find bin corresponding to angle
	# 	c. Place magnitude in bin found in previous step
	# 5. For each cell
	# 	a. For each bin add magnitudes in that bin for all pixels in cell (hog_img: plotting purposes)
	# 6. Combine cells in blocks (blocks have one cell overlapping in both directions) 
	# 7. For each block concatenate bin vectors of cells contained in it (hog_array: HOG features) 
    imgs = imgs.reshape(-1,640,640)
    hogs = list()
    for img in imgs:
        hog_array = hog( 
        img, 
        pixels_per_cell=(16,16),
        cells_per_block=(2,2), 
        orientations=9, 
        visualize=False,
        block_norm='L2-Hys')
        hogs.append(hog_array)
    return np.array(hogs)

## Data treatment

def split_data(X,y):
    return train_test_split(
        X, y,
        test_size=0.3, 
        shuffle = True, 
        random_state = 0,
        stratify = y
    )

def load_data(input_path,crack_path,pothole_path):
	X = np.load(open(input_path,'rb'))
	crack = np.load(open(crack_path,'rb'))
	pothole = np.load(open(pothole_path,'rb'))
	return X, crack, pothole

def load_models(svmcrack_path,svmhole_path,crackpipe_path,holepipe_path,enscrack_path,enshole_path):
    svmcrack = dill.load(open(svmcrack_path,'rb')) 
    svmhole = dill.load(open(svmhole_path,'rb')) 
    crackpipe = dill.load(open(crackpipe_path,'rb'))
    holepipe = dill.load(open(holepipe_path,'rb'))
    enscrack = dill.load(open(enscrack_path,'rb'))
    enshole = dill.load(open(enshole_path,'rb'))
    return svmcrack, svmhole, crackpipe, holepipe, enscrack, enshole

## Configuration

class Config():
	def __init__(self,config_file="config.yaml"):
		self.config_file = config_file
	def get_config(self):
		return yaml.load(open(self.config_file, "r"), Loader=yaml.FullLoader)
