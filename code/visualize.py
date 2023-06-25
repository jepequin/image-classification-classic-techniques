import os
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from utils import Config

args = Config().get_config()

data_dir = args['data_dir']
result_dir = args['result_dir']
img_dir = args['img_dir']

# Plot hog, masks, and original 
fig, axs = plt.subplots(1,4)
img = imread(os.path.join(data_dir,img_dir,img_dir + '_RAW.jpg'))
crack_mask = imread(os.path.join(data_dir,img_dir,img_dir + '_CRACK.png'))
pothole_mask = imread(os.path.join(data_dir,img_dir,img_dir + '_POTHOLE.png'))
img = resize(img,(640,640),anti_aliasing=True)
crack_mask = resize(crack_mask,(640,640),anti_aliasing=True)
pothole_mask = resize(pothole_mask,(640,640),anti_aliasing=True)
_, hog_img = hog( 
	rgb2gray(img), 
	pixels_per_cell=(16,16),
	cells_per_block=(2,2), 
	orientations=9, 
	visualize=True,
	block_norm='L2-Hys'
	) 
axs[0].imshow(crack_mask)
axs[0].set_xlabel('Crack mask')
axs[1].imshow(pothole_mask)
axs[1].set_xlabel('Pothole mask')
axs[2].imshow(img)
axs[2].set_xlabel('Original image')
axs[3].imshow(hog_img)
axs[3].set_xlabel('Hog image')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig(os.path.join(result_dir,img_dir + '.png'))

