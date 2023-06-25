import os
import pandas as pd

from skimage.io import imread

data_dir = '../data/images'
defects = ['crack','pothole']
label_dic = {
	# 'image_path': [],
	'image_name': [],
	'crack': [],
	'pothole':[]
}

for subdir in os.listdir(data_dir):
	image_dir = os.path.join(data_dir,subdir)
	# raw_image_name = image_dir + '_RAW.jpg'
	# raw_image_path = os.path.join(image_dir,raw_image_name)
	# label_dic['image_path'].append(raw_image_path) 
	label_dic['image_name'].append(subdir)
	for image_name in os.listdir(image_dir):
		for defect in defects:
			if defect in image_name.lower():
				image_path = os.path.join(image_dir,image_name)
				image = imread(image_path,as_gray=True)
				label_dic[defect].append(bool(image.sum()))

label_df = pd.DataFrame.from_dict(label_dic)		
label_df.to_csv(os.path.join(data_dir,'labels.csv'))
print("Labels saved to 'labels.csv' in 'data/images' directory")