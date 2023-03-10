import os
import sys
from glob import glob
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base.keras_data_loader_base import DataLoader
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import load_img
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

class LasDataLoader(DataLoader):
	def __init__(self, config):
		super().__init__(config)
		return

	def load_dataset(self):
		print("Loading the dataset from local directory")
		
		# img_paths, label_paths = glob(self.config.data_path)
		img_paths = glob(os.path.join(self.config.config_namespace.DATA_PATH, '*.png'))
		img_paths = img_paths[:self.config.config_namespace.DATA_NUM]
		
		imgs = []
		for img_path in tqdm(img_paths, desc="Load original images...", ascii=True, leave=True, colour='red'):
			img = load_img(img_path)		# load_img return PIL object
			img = np.array(img)
			imgs.append(img)
		self.train_x, self.train_y = np.array(imgs), None		# Batch * Height * Width * Channel
																# self.train_y will be assigned from preprocess_dataset method


	def print_dataset_details(self):
		print(f"Total Data Shape: {self.train_x.shape}")
		print(f"Total train_y Data Shape: {self.train_y.shape}")

	def preprocess_dataset(self):

		# Numpy Broadcasting - Normalization
		# self.train_x = self.train_x / 255.0

		# visualize data(But This can't operate on normalized images)
		# im1 = Image.fromarray(self.train_x[0])
		# im1.save("im1.jpg")
		
		imgs=[]
		for img in tqdm(self.train_x, desc="Denoising to make train_y data for autoencoder...", ascii=True, colour='green'):
			binary_img = np.where(img < 120, 0, img)
			imgs.append(binary_img)
		self.train_y = np.array(imgs)

if __name__ == '__main__':
	with open('cfg/las_cfg.yaml') as f:
		conf = yaml.load(f, Loader=yaml.FullLoader)
		print(conf)
	# dataloader = LasDataLoader()