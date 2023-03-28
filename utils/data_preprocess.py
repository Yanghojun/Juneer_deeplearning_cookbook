from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil
import sys
from tqdm import tqdm

def data_split(data_dir:str):
	img_paths = glob(os.path.join(data_dir, '*.png'))
	txt_paths = glob(os.path.join(data_dir, '*.txt'))

	# delete 'classes.txt' path
	tp_paths = []
	for path in txt_paths:
		file_name = os.path.split(path)[-1]
		if file_name != 'classes.txt':
			tp_paths.append(path)
	txt_paths = tp_paths

	img_paths.sort()
	txt_paths.sort()

	if len(txt_paths) != 0:
		train_img_paths, test_img_paths, train_txt_paths, test_txt_paths = train_test_split(img_paths, txt_paths, random_state=5, test_size=0.2)
	else:
		train_img_paths, test_img_paths = train_test_split(img_paths, random_state=5, test_size=0.2)
	
	train_dir = os.path.join(data_dir, 'train')
	test_dir = os.path.join(data_dir, 'test')

	os.makedirs(train_dir, exist_ok=True), os.makedirs(test_dir, exist_ok=True)

	cnt = 0
	
	for path in train_img_paths:
		file_name = os.path.split(path)[-1]
		shutil.move(path, os.path.join(train_dir, file_name))

	for path in test_img_paths:
		file_name = os.path.split(path)[-1]
		shutil.move(path, os.path.join(test_dir, file_name))

	if len(txt_paths) != 0:
		for path in train_txt_paths:
			file_name = os.path.split(path)[-1]
			shutil.move(path, os.path.join(train_dir, file_name))

		for path in test_txt_paths:
			file_name = os.path.split(path)[-1]
			shutil.move(path, os.path.join(test_dir, file_name))

LABEL_KEWORD='_denoised'
ORI_PATH='./dataset/las_data_annotated_v3_23_03_03'
def tp(path:str):
	label_paths = glob(path + '/**/*.png')
	ori_whole_img_paths = glob(ORI_PATH+'/**/*.png')

	for label_path in tqdm(label_paths):
		label_dir, label_file_name = os.path.split(label_path)
		label_file_name = label_file_name.replace(LABEL_KEWORD, '')
		
		for img_path in ori_whole_img_paths:
			_, img_file_name = os.path.split(img_path)
			if img_file_name == label_file_name:
				print(img_path, label_dir + '/' + img_file_name)
				shutil.copyfile(img_path, label_dir + '/' + img_file_name)
				break
		

if __name__ == '__main__':
	# data_split('./dataset/las_data_hand_denoised/')
	tp('./dataset/las_data_hand_denoised')